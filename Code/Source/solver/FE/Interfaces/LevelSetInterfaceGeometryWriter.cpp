/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Interfaces/LevelSetInterfaceGeometryWriter.h"

#include <array>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace interfaces {
namespace {

struct InterfaceGeometryVtpData {
    std::vector<std::array<Real, 3>> points{};
    std::vector<std::array<Real, 3>> point_parent_coordinates{};
    std::vector<Real> point_level_set_values{};
    std::vector<Real> point_level_set_residuals{};
    std::vector<std::int64_t> point_quadrature_flags{};
    std::vector<std::size_t> vertex_connectivity{};
    std::vector<std::size_t> vertex_offsets{};
    std::vector<std::array<Real, 3>> vertex_normals{};
    std::vector<Real> vertex_curvature_estimates{};
    std::vector<Real> vertex_negative_volume_fractions{};
    std::vector<Real> vertex_positive_volume_fractions{};
    std::vector<std::int64_t> vertex_interface_markers{};
    std::vector<std::int64_t> vertex_parent_cells{};
    std::vector<std::int64_t> vertex_fragment_stable_ids{};
    std::vector<std::int64_t> vertex_fragment_kind_ids{};
    std::vector<std::int64_t> vertex_curved_patch_flags{};
    std::vector<std::int64_t> vertex_local_fragment_indices{};
    std::vector<std::int64_t> vertex_conditioning_status_ids{};
    std::vector<std::int64_t> vertex_requested_quadrature_orders{};
    std::vector<std::int64_t> vertex_achieved_quadrature_orders{};
    std::vector<std::int64_t> vertex_fallback_status_ids{};
    std::vector<std::int64_t> vertex_root_polished_flags{};
    std::vector<std::int64_t> vertex_root_finder_iterations{};
    std::vector<Real> vertex_max_root_residuals{};
    std::vector<Real> vertex_min_gradient_norms{};
    std::vector<std::int64_t> vertex_quadrature_point_flags{};
    std::vector<Real> vertex_quadrature_weights{};
    std::vector<Real> vertex_reference_measure_factors{};
    std::vector<Real> vertex_level_set_residuals{};
    std::vector<Real> vertex_gradient_norms{};
    std::vector<std::size_t> line_connectivity{};
    std::vector<std::size_t> line_offsets{};
    std::vector<std::array<Real, 3>> line_normals{};
    std::vector<Real> line_curvature_estimates{};
    std::vector<Real> line_negative_volume_fractions{};
    std::vector<Real> line_positive_volume_fractions{};
    std::vector<std::int64_t> line_interface_markers{};
    std::vector<std::int64_t> line_parent_cells{};
    std::vector<std::int64_t> line_fragment_stable_ids{};
    std::vector<std::int64_t> line_fragment_kind_ids{};
    std::vector<std::int64_t> line_curved_patch_flags{};
    std::vector<std::int64_t> line_local_fragment_indices{};
    std::vector<std::int64_t> line_conditioning_status_ids{};
    std::vector<std::int64_t> line_requested_quadrature_orders{};
    std::vector<std::int64_t> line_achieved_quadrature_orders{};
    std::vector<std::int64_t> line_fallback_status_ids{};
    std::vector<std::int64_t> line_root_polished_flags{};
    std::vector<std::int64_t> line_root_finder_iterations{};
    std::vector<Real> line_max_root_residuals{};
    std::vector<Real> line_min_gradient_norms{};
    std::vector<std::int64_t> line_quadrature_point_flags{};
    std::vector<Real> line_quadrature_weights{};
    std::vector<Real> line_reference_measure_factors{};
    std::vector<Real> line_level_set_residuals{};
    std::vector<Real> line_gradient_norms{};
    std::vector<std::size_t> polygon_connectivity{};
    std::vector<std::size_t> polygon_offsets{};
    std::vector<std::array<Real, 3>> polygon_normals{};
    std::vector<Real> polygon_curvature_estimates{};
    std::vector<Real> polygon_negative_volume_fractions{};
    std::vector<Real> polygon_positive_volume_fractions{};
    std::vector<std::int64_t> polygon_interface_markers{};
    std::vector<std::int64_t> polygon_parent_cells{};
    std::vector<std::int64_t> polygon_fragment_stable_ids{};
    std::vector<std::int64_t> polygon_fragment_kind_ids{};
    std::vector<std::int64_t> polygon_curved_patch_flags{};
    std::vector<std::int64_t> polygon_local_fragment_indices{};
    std::vector<std::int64_t> polygon_conditioning_status_ids{};
    std::vector<std::int64_t> polygon_requested_quadrature_orders{};
    std::vector<std::int64_t> polygon_achieved_quadrature_orders{};
    std::vector<std::int64_t> polygon_fallback_status_ids{};
    std::vector<std::int64_t> polygon_root_polished_flags{};
    std::vector<std::int64_t> polygon_root_finder_iterations{};
    std::vector<Real> polygon_max_root_residuals{};
    std::vector<Real> polygon_min_gradient_norms{};
    std::vector<std::int64_t> polygon_quadrature_point_flags{};
    std::vector<Real> polygon_quadrature_weights{};
    std::vector<Real> polygon_reference_measure_factors{};
    std::vector<Real> polygon_level_set_residuals{};
    std::vector<Real> polygon_gradient_norms{};
};

[[nodiscard]] std::int64_t signedStableId(std::uint64_t id) noexcept
{
    constexpr auto max_signed =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    return id > max_signed ? std::numeric_limits<std::int64_t>::max()
                           : static_cast<std::int64_t>(id);
}

[[nodiscard]] std::int64_t signedLocalIndex(LocalIndex index) noexcept
{
    return index == INVALID_LOCAL_INDEX ? std::int64_t{-1}
                                        : static_cast<std::int64_t>(index);
}

[[nodiscard]] std::int64_t fragmentKindId(
    CutInterfaceFragmentKind kind) noexcept
{
    switch (kind) {
    case CutInterfaceFragmentKind::Segment:
        return 0;
    case CutInterfaceFragmentKind::Polygon:
        return 1;
    case CutInterfaceFragmentKind::CurvedPatch:
        return 2;
    }
    return -1;
}

[[nodiscard]] std::int64_t conditioningStatusId(
    const std::string& diagnostic) noexcept
{
    if (diagnostic.empty()) {
        return 0;
    }
    return diagnostic == "ok" ? 1 : 2;
}

[[nodiscard]] std::int64_t fallbackStatusId(
    const std::string& status) noexcept
{
    if (status.empty() || status == "Unknown") {
        return 0;
    }
    if (status == "None") {
        return 1;
    }
    if (status == "Used") {
        return 2;
    }
    return 3;
}

[[nodiscard]] const std::string& fragmentFallbackStatus(
    const CutInterfaceFragment& fragment,
    const CutInterfaceDomainRequest& request) noexcept
{
    return fragment.implicit_fallback_status.empty()
               ? request.implicit_fallback_status
               : fragment.implicit_fallback_status;
}

[[nodiscard]] std::int64_t achievedInterfaceQuadratureOrder(
    const CutInterfaceFragment& fragment,
    const CutInterfaceDomainRequest& request) noexcept
{
    try {
        return fragment.toCutQuadratureRule(request)
            .provenance.achieved_quadrature_order;
    } catch (...) {
        return -1;
    }
}

void appendFragmentCellData(const CutInterfaceFragment& fragment,
                            const std::array<Real, 3>& cell_normal,
                            std::vector<std::array<Real, 3>>& normals,
                            std::vector<Real>& curvature_estimates,
                            std::vector<Real>& negative_volume_fractions,
                            std::vector<Real>& positive_volume_fractions,
                            std::vector<std::int64_t>& interface_markers,
                            std::vector<std::int64_t>& parent_cells,
                            std::vector<std::int64_t>& fragment_stable_ids,
                            std::vector<std::int64_t>& fragment_kind_ids,
                            std::vector<std::int64_t>& curved_patch_flags,
                            std::vector<std::int64_t>& local_fragment_indices,
                            std::vector<std::int64_t>& conditioning_status_ids,
                            std::vector<std::int64_t>& requested_quadrature_orders,
                            std::vector<std::int64_t>& achieved_quadrature_orders,
                            std::vector<std::int64_t>& fallback_status_ids,
                            std::vector<std::int64_t>& root_polished_flags,
                            std::vector<std::int64_t>& root_finder_iterations,
                            std::vector<Real>& max_root_residuals,
                            std::vector<Real>& min_gradient_norms,
                            std::vector<std::int64_t>& quadrature_point_flags,
                            std::vector<Real>& quadrature_weights,
                            std::vector<Real>& reference_measure_factors,
                            std::vector<Real>& level_set_residuals,
                            std::vector<Real>& gradient_norms,
                            bool quadrature_point_cell,
                            Real quadrature_weight,
                            Real reference_measure_factor,
                            Real level_set_residual,
                            Real gradient_norm,
                            std::int64_t requested_quadrature_order,
                            std::int64_t achieved_quadrature_order,
                            std::int64_t fallback_status_id)
{
    normals.push_back(cell_normal);
    curvature_estimates.push_back(fragment.curvature_estimate);
    negative_volume_fractions.push_back(fragment.negative_volume_fraction);
    positive_volume_fractions.push_back(fragment.positive_volume_fraction);
    interface_markers.push_back(static_cast<std::int64_t>(fragment.interface_marker));
    parent_cells.push_back(static_cast<std::int64_t>(fragment.parent_cell));
    fragment_stable_ids.push_back(signedStableId(fragment.stable_id));
    fragment_kind_ids.push_back(fragmentKindId(fragment.kind));
    curved_patch_flags.push_back(
        fragment.kind == CutInterfaceFragmentKind::CurvedPatch ? 1 : 0);
    local_fragment_indices.push_back(
        signedLocalIndex(fragment.local_fragment_index));
    conditioning_status_ids.push_back(
        conditioningStatusId(fragment.conditioning_diagnostic));
    requested_quadrature_orders.push_back(requested_quadrature_order);
    achieved_quadrature_orders.push_back(achieved_quadrature_order);
    fallback_status_ids.push_back(fallback_status_id);
    root_polished_flags.push_back(fragment.root_polished ? 1 : 0);
    root_finder_iterations.push_back(
        static_cast<std::int64_t>(fragment.root_finder_iterations));
    max_root_residuals.push_back(fragment.max_root_residual);
    min_gradient_norms.push_back(fragment.min_gradient_norm);
    quadrature_point_flags.push_back(quadrature_point_cell ? 1 : 0);
    quadrature_weights.push_back(quadrature_weight);
    reference_measure_factors.push_back(reference_measure_factor);
    level_set_residuals.push_back(level_set_residual);
    gradient_norms.push_back(gradient_norm);
}

[[nodiscard]] bool writeFragmentAsLineCell(
    const CutInterfaceFragment& fragment) noexcept
{
    return fragment.kind == CutInterfaceFragmentKind::Segment ||
           fragment.vertices.size() == 2u;
}

[[nodiscard]] InterfaceGeometryVtpData collectVtpData(
    const LevelSetInterfaceDomain& domain)
{
    InterfaceGeometryVtpData data;
    for (const auto& fragment : domain.fragments()) {
        if (!fragment.active() || fragment.vertices.empty()) {
            continue;
        }
        const std::size_t first_point = data.points.size();
        const std::int64_t requested_quadrature_order =
            domain.request().resolvedInterfaceQuadratureOrder();
        const std::int64_t achieved_quadrature_order =
            achievedInterfaceQuadratureOrder(fragment, domain.request());
        const std::int64_t fallback_status_id =
            fallbackStatusId(fragmentFallbackStatus(fragment, domain.request()));
        for (const auto& vertex : fragment.vertices) {
            data.points.push_back(vertex.point);
            data.point_parent_coordinates.push_back(vertex.parent_coordinate);
            data.point_level_set_values.push_back(vertex.level_set_value);
            data.point_level_set_residuals.push_back(vertex.level_set_value);
            data.point_quadrature_flags.push_back(0);
        }

        if (writeFragmentAsLineCell(fragment)) {
            for (std::size_t i = 0; i < fragment.vertices.size(); ++i) {
                data.line_connectivity.push_back(first_point + i);
            }
            data.line_offsets.push_back(data.line_connectivity.size());
            appendFragmentCellData(fragment,
                                   fragment.normal,
                                   data.line_normals,
                                   data.line_curvature_estimates,
                                   data.line_negative_volume_fractions,
                                   data.line_positive_volume_fractions,
                                   data.line_interface_markers,
                                   data.line_parent_cells,
                                   data.line_fragment_stable_ids,
                                   data.line_fragment_kind_ids,
                                   data.line_curved_patch_flags,
                                   data.line_local_fragment_indices,
                                   data.line_conditioning_status_ids,
                                   data.line_requested_quadrature_orders,
                                   data.line_achieved_quadrature_orders,
                                   data.line_fallback_status_ids,
                                   data.line_root_polished_flags,
                                   data.line_root_finder_iterations,
                                   data.line_max_root_residuals,
                                   data.line_min_gradient_norms,
                                   data.line_quadrature_point_flags,
                                   data.line_quadrature_weights,
                                   data.line_reference_measure_factors,
                                   data.line_level_set_residuals,
                                   data.line_gradient_norms,
                                   /*quadrature_point_cell=*/false,
                                   /*quadrature_weight=*/Real{0.0},
                                   /*reference_measure_factor=*/fragment.measure,
                                   fragment.max_root_residual,
                                   fragment.min_gradient_norm,
                                   requested_quadrature_order,
                                   achieved_quadrature_order,
                                   fallback_status_id);
        } else {
            for (std::size_t i = 0; i < fragment.vertices.size(); ++i) {
                data.polygon_connectivity.push_back(first_point + i);
            }
            data.polygon_offsets.push_back(data.polygon_connectivity.size());
            appendFragmentCellData(fragment,
                                   fragment.normal,
                                   data.polygon_normals,
                                   data.polygon_curvature_estimates,
                                   data.polygon_negative_volume_fractions,
                                   data.polygon_positive_volume_fractions,
                                   data.polygon_interface_markers,
                                   data.polygon_parent_cells,
                                   data.polygon_fragment_stable_ids,
                                   data.polygon_fragment_kind_ids,
                                   data.polygon_curved_patch_flags,
                                   data.polygon_local_fragment_indices,
                                   data.polygon_conditioning_status_ids,
                                   data.polygon_requested_quadrature_orders,
                                   data.polygon_achieved_quadrature_orders,
                                   data.polygon_fallback_status_ids,
                                   data.polygon_root_polished_flags,
                                   data.polygon_root_finder_iterations,
                                   data.polygon_max_root_residuals,
                                   data.polygon_min_gradient_norms,
                                   data.polygon_quadrature_point_flags,
                                   data.polygon_quadrature_weights,
                                   data.polygon_reference_measure_factors,
                                   data.polygon_level_set_residuals,
                                   data.polygon_gradient_norms,
                                   /*quadrature_point_cell=*/false,
                                   /*quadrature_weight=*/Real{0.0},
                                   /*reference_measure_factor=*/fragment.measure,
                                   fragment.max_root_residual,
                                   fragment.min_gradient_norm,
                                   requested_quadrature_order,
                                   achieved_quadrature_order,
                                   fallback_status_id);
        }

        for (const auto& qpoint : fragment.quadrature_points) {
            const std::size_t point_index = data.points.size();
            data.points.push_back(qpoint.point);
            data.point_parent_coordinates.push_back(qpoint.parent_coordinate);
            data.point_level_set_values.push_back(qpoint.level_set_residual);
            data.point_level_set_residuals.push_back(qpoint.level_set_residual);
            data.point_quadrature_flags.push_back(1);
            data.vertex_connectivity.push_back(point_index);
            data.vertex_offsets.push_back(data.vertex_connectivity.size());
            appendFragmentCellData(fragment,
                                   qpoint.normal,
                                   data.vertex_normals,
                                   data.vertex_curvature_estimates,
                                   data.vertex_negative_volume_fractions,
                                   data.vertex_positive_volume_fractions,
                                   data.vertex_interface_markers,
                                   data.vertex_parent_cells,
                                   data.vertex_fragment_stable_ids,
                                   data.vertex_fragment_kind_ids,
                                   data.vertex_curved_patch_flags,
                                   data.vertex_local_fragment_indices,
                                   data.vertex_conditioning_status_ids,
                                   data.vertex_requested_quadrature_orders,
                                   data.vertex_achieved_quadrature_orders,
                                   data.vertex_fallback_status_ids,
                                   data.vertex_root_polished_flags,
                                   data.vertex_root_finder_iterations,
                                   data.vertex_max_root_residuals,
                                   data.vertex_min_gradient_norms,
                                   data.vertex_quadrature_point_flags,
                                   data.vertex_quadrature_weights,
                                   data.vertex_reference_measure_factors,
                                   data.vertex_level_set_residuals,
                                   data.vertex_gradient_norms,
                                   /*quadrature_point_cell=*/true,
                                   qpoint.weight,
                                   qpoint.reference_measure_factor,
                                   qpoint.level_set_residual,
                                   qpoint.gradient_norm,
                                   requested_quadrature_order,
                                   achieved_quadrature_order,
                                   fallback_status_id);
        }
    }
    return data;
}

void writeIndexArray(std::ostream& out,
                     const char* name,
                     const std::vector<std::size_t>& values)
{
    out << "        <DataArray type=\"Int64\" Name=\"" << name
        << "\" format=\"ascii\">";
    for (const auto value : values) {
        out << ' ' << value;
    }
    out << " </DataArray>\n";
}

void writeRealArray(std::ostream& out,
                    const char* name,
                    const std::vector<Real>& values)
{
    out << "        <DataArray type=\"Float64\" Name=\"" << name
        << "\" format=\"ascii\">";
    for (const auto value : values) {
        out << ' ' << value;
    }
    out << " </DataArray>\n";
}

void writeVectorArray(std::ostream& out,
                      const char* name,
                      const std::vector<std::array<Real, 3>>& values)
{
    out << "        <DataArray type=\"Float64\" Name=\"" << name
        << "\" NumberOfComponents=\"3\" format=\"ascii\">";
    for (const auto& value : values) {
        out << ' ' << value[0] << ' ' << value[1] << ' ' << value[2];
    }
    out << " </DataArray>\n";
}

void writeSignedIndexArray(std::ostream& out,
                           const char* name,
                           const std::vector<std::int64_t>& values)
{
    out << "        <DataArray type=\"Int64\" Name=\"" << name
        << "\" format=\"ascii\">";
    for (const auto value : values) {
        out << ' ' << value;
    }
    out << " </DataArray>\n";
}

template <typename T>
[[nodiscard]] std::vector<T> concatenate(const std::vector<T>& first,
                                         const std::vector<T>& second)
{
    std::vector<T> values;
    values.reserve(first.size() + second.size());
    values.insert(values.end(), first.begin(), first.end());
    values.insert(values.end(), second.begin(), second.end());
    return values;
}

template <typename T>
[[nodiscard]] std::vector<T> concatenate(const std::vector<T>& first,
                                         const std::vector<T>& second,
                                         const std::vector<T>& third)
{
    std::vector<T> values;
    values.reserve(first.size() + second.size() + third.size());
    values.insert(values.end(), first.begin(), first.end());
    values.insert(values.end(), second.begin(), second.end());
    values.insert(values.end(), third.begin(), third.end());
    return values;
}

[[nodiscard]] const char* sideName(geometry::CutIntegrationSide side) noexcept
{
    switch (side) {
    case geometry::CutIntegrationSide::Negative:
        return "negative";
    case geometry::CutIntegrationSide::Positive:
        return "positive";
    case geometry::CutIntegrationSide::Interface:
        return "interface";
    }
    return "unknown";
}

[[nodiscard]] const char* quadratureKindName(geometry::CutQuadratureKind kind) noexcept
{
    switch (kind) {
    case geometry::CutQuadratureKind::Volume:
        return "volume";
    case geometry::CutQuadratureKind::Face:
        return "face";
    case geometry::CutQuadratureKind::Interface:
        return "interface";
    }
    return "unknown";
}

[[nodiscard]] const char* constructionKindName(
    geometry::CutQuadratureConstructionKind kind) noexcept
{
    switch (kind) {
    case geometry::CutQuadratureConstructionKind::AxisAlignedBoxClip:
        return "axis_aligned_box_clip";
    case geometry::CutQuadratureConstructionKind::SegmentClip:
        return "segment_clip";
    case geometry::CutQuadratureConstructionKind::PolygonInterface:
        return "polygon_interface";
    case geometry::CutQuadratureConstructionKind::ConservativeMomentFit:
        return "conservative_moment_fit";
    case geometry::CutQuadratureConstructionKind::TopologySubdivision:
        return "topology_subdivision";
    case geometry::CutQuadratureConstructionKind::CurvedInterface:
        return "curved_interface";
    case geometry::CutQuadratureConstructionKind::MomentFittedImplicit:
        return "moment_fitted_implicit";
    case geometry::CutQuadratureConstructionKind::CurvedTopologySubdivision:
        return "curved_topology_subdivision";
    }
    return "unknown";
}

[[nodiscard]] const char* frameName(geometry::CutGeometryFrame frame) noexcept
{
    switch (frame) {
    case geometry::CutGeometryFrame::Reference:
        return "reference";
    case geometry::CutGeometryFrame::Current:
        return "current";
    }
    return "unknown";
}

[[nodiscard]] const char* sourceKindName(CutInterfaceSourceKind kind) noexcept
{
    switch (kind) {
    case CutInterfaceSourceKind::Field:
        return "field";
    case CutInterfaceSourceKind::Evaluator:
        return "evaluator";
    }
    return "unknown";
}

void writeJsonString(std::ostream& out, const std::string& value)
{
    out << '"';
    for (const char c : value) {
        switch (c) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            out << c;
            break;
        }
    }
    out << '"';
}

void writeJsonArray3(std::ostream& out, const std::array<Real, 3>& value)
{
    out << '[' << value[0] << ", " << value[1] << ", " << value[2] << ']';
}

void writeQuadratureRuleJson(const geometry::CutQuadratureRule& rule,
                             std::ostream& out,
                             const char* indent)
{
    out << indent << "{\n";
    out << indent << "  \"kind\": \"" << quadratureKindName(rule.kind) << "\",\n";
    out << indent << "  \"side\": \"" << sideName(rule.side) << "\",\n";
    out << indent << "  \"measure\": " << rule.measure << ",\n";
    out << indent << "  \"parent_measure\": " << rule.parent_measure << ",\n";
    out << indent << "  \"volume_fraction\": " << rule.volume_fraction << ",\n";
    out << indent << "  \"exact_for_constants\": "
        << (rule.exact_for_constants ? "true" : "false") << ",\n";
    out << indent << "  \"exact_polynomial_order\": "
        << rule.exact_polynomial_order << ",\n";
    out << indent << "  \"policy\": {\n";
    out << indent << "    \"name\": ";
    writeJsonString(out, rule.policy.name);
    out << ",\n";
    out << indent << "    \"construction\": \""
        << constructionKindName(rule.policy.kind) << "\",\n";
    out << indent << "    \"polynomial_order\": "
        << rule.policy.polynomial_order << ",\n";
    out << indent << "    \"moment_fitted\": "
        << (rule.policy.moment_fitted ? "true" : "false") << ",\n";
    out << indent << "    \"tolerance\": " << rule.policy.tolerance << '\n';
    out << indent << "  },\n";
    out << indent << "  \"provenance\": {\n";
    out << indent << "    \"embedded_geometry_id\": ";
    writeJsonString(out, rule.provenance.embedded_geometry_id);
    out << ",\n";
    out << indent << "    \"cut_topology_id\": ";
    writeJsonString(out, rule.provenance.cut_topology_id);
    out << ",\n";
    out << indent << "    \"parent_entity\": "
        << rule.provenance.parent_entity << ",\n";
    out << indent << "    \"marker\": " << rule.provenance.marker << ",\n";
    out << indent << "    \"cut_topology_revision\": "
        << rule.provenance.cut_topology_revision << ",\n";
    out << indent << "    \"predicate_policy_key\": "
        << rule.provenance.predicate_policy_key << ",\n";
    out << indent << "    \"construction\": \""
        << constructionKindName(rule.provenance.construction) << "\",\n";
    out << indent << "    \"frame\": \"" << frameName(rule.provenance.frame)
        << "\",\n";
    out << indent << "    \"implicit_geometry_mode\": ";
    writeJsonString(out, rule.provenance.implicit_geometry_mode);
    out << ",\n";
    out << indent << "    \"implicit_quadrature_backend\": ";
    writeJsonString(out, rule.provenance.implicit_quadrature_backend);
    out << ",\n";
    out << indent << "    \"selected_implicit_quadrature_backend\": ";
    writeJsonString(out, rule.provenance.selected_implicit_quadrature_backend);
    out << ",\n";
    out << indent << "    \"implicit_fallback_policy\": ";
    writeJsonString(out, rule.provenance.implicit_fallback_policy);
    out << ",\n";
    out << indent << "    \"implicit_fallback_status\": ";
    writeJsonString(out, rule.provenance.implicit_fallback_status);
    out << ",\n";
    out << indent << "    \"geometry_tangent_policy\": ";
    writeJsonString(out, rule.provenance.geometry_tangent_policy);
    out << ",\n";
    out << indent << "    \"implicit_cut_root_tolerance\": "
        << rule.provenance.implicit_cut_root_tolerance << ",\n";
    out << indent << "    \"implicit_cut_root_coordinate_tolerance\": "
        << rule.provenance.implicit_cut_root_coordinate_tolerance << ",\n";
    out << indent << "    \"implicit_cut_root_max_iterations\": "
        << rule.provenance.implicit_cut_root_max_iterations << ",\n";
    out << indent << "    \"requested_quadrature_order\": "
        << rule.provenance.requested_quadrature_order << ",\n";
    out << indent << "    \"achieved_quadrature_order\": "
        << rule.provenance.achieved_quadrature_order << '\n';
    out << indent << "  },\n";
    out << indent << "  \"provenance_id\": ";
    writeJsonString(out, rule.provenance_id);
    out << ",\n";
    out << indent << "  \"frame\": \"" << frameName(rule.frame) << "\",\n";
    out << indent << "  \"curved_geometry\": "
        << (rule.curved_geometry ? "true" : "false") << ",\n";
    out << indent << "  \"full_cell_equivalent\": "
        << (rule.full_cell_equivalent ? "true" : "false") << ",\n";
    out << indent << "  \"quadrature_points\": [";
    if (!rule.points.empty()) {
        out << '\n';
        for (std::size_t i = 0; i < rule.points.size(); ++i) {
            const auto& point = rule.points[i];
            out << indent << "    {\n";
            out << indent << "      \"point\": ";
            writeJsonArray3(out, point.point);
            out << ",\n";
            out << indent << "      \"parent_coordinate\": ";
            writeJsonArray3(out, point.parent_coordinate);
            out << ",\n";
            out << indent << "      \"normal\": ";
            writeJsonArray3(out, point.normal);
            out << ",\n";
            out << indent << "      \"weight\": " << point.weight << ",\n";
            out << indent << "      \"reference_measure_factor\": "
                << point.reference_measure_factor << ",\n";
            out << indent << "      \"level_set_residual\": "
                << point.level_set_residual << ",\n";
            out << indent << "      \"gradient_norm\": "
                << point.gradient_norm << '\n';
            out << indent << "    }";
            if (i + 1 < rule.points.size()) {
                out << ',';
            }
            out << '\n';
        }
        out << indent << "  ";
    }
    out << "]\n";
    out << indent << '}';
}

void writeRuleListJson(const std::vector<geometry::CutQuadratureRule>& rules,
                       MeshIndex parent_cell,
                       std::ostream& out,
                       const char* indent)
{
    out << "[";
    bool wrote_any = false;
    for (const auto& rule : rules) {
        if (rule.provenance.parent_entity != parent_cell) {
            continue;
        }
        out << (wrote_any ? ",\n" : "\n");
        writeQuadratureRuleJson(rule, out, indent);
        wrote_any = true;
    }
    if (wrote_any) {
        out << '\n' << indent;
    }
    out << "]";
}

} // namespace

void writeLevelSetInterfaceGeometryVtp(const LevelSetInterfaceDomain& domain,
                                       std::ostream& out)
{
    const auto data = collectVtpData(domain);
    out << std::setprecision(17);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <PolyData>\n";
    out << "    <Piece NumberOfPoints=\"" << data.points.size()
        << "\" NumberOfVerts=\"" << data.vertex_offsets.size()
        << "\" NumberOfLines=\"" << data.line_offsets.size()
        << "\" NumberOfPolys=\"" << data.polygon_offsets.size() << "\">\n";
    out << "      <PointData>\n";
    writeRealArray(out, "level_set_value", data.point_level_set_values);
    writeRealArray(out, "level_set_residual", data.point_level_set_residuals);
    writeVectorArray(out, "parent_coordinate", data.point_parent_coordinates);
    writeSignedIndexArray(out, "quadrature_point", data.point_quadrature_flags);
    out << "      </PointData>\n";
    out << "      <CellData>\n";
    writeVectorArray(out,
                     "interface_normal",
                     concatenate(data.vertex_normals,
                                 data.line_normals,
                                 data.polygon_normals));
    writeRealArray(out,
                   "curvature_estimate",
                   concatenate(data.vertex_curvature_estimates,
                               data.line_curvature_estimates,
                               data.polygon_curvature_estimates));
    writeRealArray(out,
                   "negative_volume_fraction",
                   concatenate(data.vertex_negative_volume_fractions,
                               data.line_negative_volume_fractions,
                               data.polygon_negative_volume_fractions));
    writeRealArray(out,
                   "positive_volume_fraction",
                   concatenate(data.vertex_positive_volume_fractions,
                               data.line_positive_volume_fractions,
                               data.polygon_positive_volume_fractions));
    writeSignedIndexArray(out,
                          "interface_marker",
                          concatenate(data.vertex_interface_markers,
                                      data.line_interface_markers,
                                      data.polygon_interface_markers));
    writeSignedIndexArray(out,
                          "parent_cell",
                          concatenate(data.vertex_parent_cells,
                                      data.line_parent_cells,
                                      data.polygon_parent_cells));
    writeSignedIndexArray(out,
                          "fragment_stable_id",
                          concatenate(data.vertex_fragment_stable_ids,
                                      data.line_fragment_stable_ids,
                                      data.polygon_fragment_stable_ids));
    writeSignedIndexArray(out,
                          "fragment_kind",
                          concatenate(data.vertex_fragment_kind_ids,
                                      data.line_fragment_kind_ids,
                                      data.polygon_fragment_kind_ids));
    writeSignedIndexArray(out,
                          "curved_patch",
                          concatenate(data.vertex_curved_patch_flags,
                                      data.line_curved_patch_flags,
                                      data.polygon_curved_patch_flags));
    writeSignedIndexArray(out,
                          "local_fragment_index",
                          concatenate(data.vertex_local_fragment_indices,
                                      data.line_local_fragment_indices,
                                      data.polygon_local_fragment_indices));
    writeSignedIndexArray(out,
                          "conditioning_status",
                          concatenate(data.vertex_conditioning_status_ids,
                                      data.line_conditioning_status_ids,
                                      data.polygon_conditioning_status_ids));
    writeSignedIndexArray(out,
                          "requested_quadrature_order",
                          concatenate(data.vertex_requested_quadrature_orders,
                                      data.line_requested_quadrature_orders,
                                      data.polygon_requested_quadrature_orders));
    writeSignedIndexArray(out,
                          "achieved_quadrature_order",
                          concatenate(data.vertex_achieved_quadrature_orders,
                                      data.line_achieved_quadrature_orders,
                                      data.polygon_achieved_quadrature_orders));
    writeSignedIndexArray(out,
                          "fallback_status",
                          concatenate(data.vertex_fallback_status_ids,
                                      data.line_fallback_status_ids,
                                      data.polygon_fallback_status_ids));
    writeSignedIndexArray(out,
                          "root_polished",
                          concatenate(data.vertex_root_polished_flags,
                                      data.line_root_polished_flags,
                                      data.polygon_root_polished_flags));
    writeSignedIndexArray(out,
                          "root_finder_iterations",
                          concatenate(data.vertex_root_finder_iterations,
                                      data.line_root_finder_iterations,
                                      data.polygon_root_finder_iterations));
    writeRealArray(out,
                   "max_root_residual",
                   concatenate(data.vertex_max_root_residuals,
                               data.line_max_root_residuals,
                               data.polygon_max_root_residuals));
    writeRealArray(out,
                   "min_gradient_norm",
                   concatenate(data.vertex_min_gradient_norms,
                               data.line_min_gradient_norms,
                               data.polygon_min_gradient_norms));
    writeSignedIndexArray(out,
                          "quadrature_point",
                          concatenate(data.vertex_quadrature_point_flags,
                                      data.line_quadrature_point_flags,
                                      data.polygon_quadrature_point_flags));
    writeRealArray(out,
                   "quadrature_weight",
                   concatenate(data.vertex_quadrature_weights,
                               data.line_quadrature_weights,
                               data.polygon_quadrature_weights));
    writeRealArray(out,
                   "reference_measure_factor",
                   concatenate(data.vertex_reference_measure_factors,
                               data.line_reference_measure_factors,
                               data.polygon_reference_measure_factors));
    writeRealArray(out,
                   "quadrature_level_set_residual",
                   concatenate(data.vertex_level_set_residuals,
                               data.line_level_set_residuals,
                               data.polygon_level_set_residuals));
    writeRealArray(out,
                   "quadrature_gradient_norm",
                   concatenate(data.vertex_gradient_norms,
                               data.line_gradient_norms,
                               data.polygon_gradient_norms));
    out << "      </CellData>\n";
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">";
    for (const auto& point : data.points) {
        out << ' ' << point[0] << ' ' << point[1] << ' ' << point[2];
    }
    out << " </DataArray>\n";
    out << "      </Points>\n";
    out << "      <Verts>\n";
    writeIndexArray(out, "connectivity", data.vertex_connectivity);
    writeIndexArray(out, "offsets", data.vertex_offsets);
    out << "      </Verts>\n";
    out << "      <Lines>\n";
    writeIndexArray(out, "connectivity", data.line_connectivity);
    writeIndexArray(out, "offsets", data.line_offsets);
    out << "      </Lines>\n";
    out << "      <Polys>\n";
    writeIndexArray(out, "connectivity", data.polygon_connectivity);
    writeIndexArray(out, "offsets", data.polygon_offsets);
    out << "      </Polys>\n";
    out << "    </Piece>\n";
    out << "  </PolyData>\n";
    out << "</VTKFile>\n";
}

std::string levelSetInterfaceGeometryVtpString(
    const LevelSetInterfaceDomain& domain)
{
    std::ostringstream out;
    writeLevelSetInterfaceGeometryVtp(domain, out);
    return out.str();
}

void writeLevelSetInterfaceQuadratureDebugJson(
    const LevelSetInterfaceDomain& domain,
    MeshIndex parent_cell,
    std::ostream& out)
{
    const auto& request = domain.request();
    const auto interface_rules = domain.interfaceQuadratureRules();
    const auto volume_rules = domain.volumeQuadratureRules();

    out << std::setprecision(17);
    out << "{\n";
    out << "  \"parent_cell\": " << parent_cell << ",\n";
    out << "  \"request\": {\n";
    out << "    \"interface_marker\": " << request.interface_marker << ",\n";
    out << "    \"isovalue\": " << request.isovalue << ",\n";
    out << "    \"tolerance\": " << request.tolerance << ",\n";
    out << "    \"quadrature_order\": " << request.quadrature_order << ",\n";
    out << "    \"interface_quadrature_order\": "
        << request.resolvedInterfaceQuadratureOrder() << ",\n";
    out << "    \"volume_quadrature_order\": "
        << request.resolvedVolumeQuadratureOrder() << ",\n";
    out << "    \"frame\": \"" << frameName(request.frame) << "\",\n";
    out << "    \"mesh_geometry_revision\": "
        << request.mesh_geometry_revision << ",\n";
    out << "    \"mesh_topology_revision\": "
        << request.mesh_topology_revision << ",\n";
    out << "    \"ownership_revision\": " << request.ownership_revision << ",\n";
    out << "    \"quadrature_policy_key\": "
        << request.quadrature_policy_key << ",\n";
    out << "    \"implicit_geometry_mode\": ";
    writeJsonString(out, request.implicit_geometry_mode);
    out << ",\n";
    out << "    \"implicit_quadrature_backend\": ";
    writeJsonString(out, request.implicit_quadrature_backend);
    out << ",\n";
    out << "    \"implicit_fallback_policy\": ";
    writeJsonString(out, request.implicit_fallback_policy);
    out << ",\n";
    out << "    \"implicit_fallback_status\": ";
    writeJsonString(out, request.implicit_fallback_status);
    out << ",\n";
    out << "    \"geometry_tangent_policy\": ";
    writeJsonString(out, request.geometry_tangent_policy);
    out << ",\n";
    out << "    \"implicit_cut_root_tolerance\": "
        << request.implicit_cut_root_tolerance << ",\n";
    out << "    \"implicit_cut_root_coordinate_tolerance\": "
        << request.implicit_cut_root_coordinate_tolerance << ",\n";
    out << "    \"implicit_cut_root_max_iterations\": "
        << request.implicit_cut_root_max_iterations << ",\n";
    out << "    \"implicit_cut_max_subdivision_depth\": "
        << request.implicit_cut_max_subdivision_depth << ",\n";
    out << "    \"achieved_interface_quadrature_order\": "
        << request.achieved_interface_quadrature_order << ",\n";
    out << "    \"achieved_volume_quadrature_order\": "
        << request.achieved_volume_quadrature_order << ",\n";
    out << "    \"source\": {\n";
    out << "      \"kind\": \"" << sourceKindName(request.source.kind) << "\",\n";
    out << "      \"identifier\": ";
    writeJsonString(out, request.source.identifier());
    out << ",\n";
    out << "      \"field_id\": " << request.source.field_id << ",\n";
    out << "      \"evaluator_id\": ";
    writeJsonString(out, request.source.evaluator_id);
    out << ",\n";
    out << "      \"layout_revision\": "
        << request.source.layout_revision << ",\n";
    out << "      \"value_revision\": "
        << request.source.value_revision << '\n';
    out << "    }\n";
    out << "  },\n";
    out << "  \"interface_rules\": ";
    writeRuleListJson(interface_rules, parent_cell, out, "    ");
    out << ",\n";
    out << "  \"volume_rules\": ";
    writeRuleListJson(volume_rules, parent_cell, out, "    ");
    out << '\n';
    out << "}\n";
}

std::string levelSetInterfaceQuadratureDebugJsonString(
    const LevelSetInterfaceDomain& domain,
    MeshIndex parent_cell)
{
    std::ostringstream out;
    writeLevelSetInterfaceQuadratureDebugJson(domain, parent_cell, out);
    return out.str();
}

} // namespace interfaces
} // namespace FE
} // namespace svmp

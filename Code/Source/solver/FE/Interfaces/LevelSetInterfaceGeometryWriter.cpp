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
#include <ostream>
#include <sstream>
#include <vector>

namespace svmp {
namespace FE {
namespace interfaces {
namespace {

struct InterfaceGeometryVtpData {
    std::vector<std::array<Real, 3>> points{};
    std::vector<Real> point_level_set_values{};
    std::vector<std::size_t> line_connectivity{};
    std::vector<std::size_t> line_offsets{};
    std::vector<std::array<Real, 3>> line_normals{};
    std::vector<Real> line_curvature_estimates{};
    std::vector<Real> line_negative_volume_fractions{};
    std::vector<Real> line_positive_volume_fractions{};
    std::vector<std::int64_t> line_interface_markers{};
    std::vector<std::int64_t> line_parent_cells{};
    std::vector<std::size_t> polygon_connectivity{};
    std::vector<std::size_t> polygon_offsets{};
    std::vector<std::array<Real, 3>> polygon_normals{};
    std::vector<Real> polygon_curvature_estimates{};
    std::vector<Real> polygon_negative_volume_fractions{};
    std::vector<Real> polygon_positive_volume_fractions{};
    std::vector<std::int64_t> polygon_interface_markers{};
    std::vector<std::int64_t> polygon_parent_cells{};
};

void appendFragmentCellData(const CutInterfaceFragment& fragment,
                            std::vector<std::array<Real, 3>>& normals,
                            std::vector<Real>& curvature_estimates,
                            std::vector<Real>& negative_volume_fractions,
                            std::vector<Real>& positive_volume_fractions,
                            std::vector<std::int64_t>& interface_markers,
                            std::vector<std::int64_t>& parent_cells)
{
    normals.push_back(fragment.normal);
    curvature_estimates.push_back(fragment.curvature_estimate);
    negative_volume_fractions.push_back(fragment.negative_volume_fraction);
    positive_volume_fractions.push_back(fragment.positive_volume_fraction);
    interface_markers.push_back(static_cast<std::int64_t>(fragment.interface_marker));
    parent_cells.push_back(static_cast<std::int64_t>(fragment.parent_cell));
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
        for (const auto& vertex : fragment.vertices) {
            data.points.push_back(vertex.point);
            data.point_level_set_values.push_back(vertex.level_set_value);
        }

        if (fragment.kind == CutInterfaceFragmentKind::Segment) {
            for (std::size_t i = 0; i < fragment.vertices.size(); ++i) {
                data.line_connectivity.push_back(first_point + i);
            }
            data.line_offsets.push_back(data.line_connectivity.size());
            appendFragmentCellData(fragment,
                                   data.line_normals,
                                   data.line_curvature_estimates,
                                   data.line_negative_volume_fractions,
                                   data.line_positive_volume_fractions,
                                   data.line_interface_markers,
                                   data.line_parent_cells);
        } else {
            for (std::size_t i = 0; i < fragment.vertices.size(); ++i) {
                data.polygon_connectivity.push_back(first_point + i);
            }
            data.polygon_offsets.push_back(data.polygon_connectivity.size());
            appendFragmentCellData(fragment,
                                   data.polygon_normals,
                                   data.polygon_curvature_estimates,
                                   data.polygon_negative_volume_fractions,
                                   data.polygon_positive_volume_fractions,
                                   data.polygon_interface_markers,
                                   data.polygon_parent_cells);
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
    out << indent << "    \"implicit_fallback_policy\": ";
    writeJsonString(out, rule.provenance.implicit_fallback_policy);
    out << ",\n";
    out << indent << "    \"geometry_tangent_policy\": ";
    writeJsonString(out, rule.provenance.geometry_tangent_policy);
    out << ",\n";
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
            out << indent << "      \"normal\": ";
            writeJsonArray3(out, point.normal);
            out << ",\n";
            out << indent << "      \"weight\": " << point.weight << '\n';
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
        << "\" NumberOfLines=\"" << data.line_offsets.size()
        << "\" NumberOfPolys=\"" << data.polygon_offsets.size() << "\">\n";
    out << "      <PointData>\n";
    writeRealArray(out, "level_set_value", data.point_level_set_values);
    out << "      </PointData>\n";
    out << "      <CellData>\n";
    writeVectorArray(out,
                     "interface_normal",
                     concatenate(data.line_normals, data.polygon_normals));
    writeRealArray(out,
                   "curvature_estimate",
                   concatenate(data.line_curvature_estimates,
                               data.polygon_curvature_estimates));
    writeRealArray(out,
                   "negative_volume_fraction",
                   concatenate(data.line_negative_volume_fractions,
                               data.polygon_negative_volume_fractions));
    writeRealArray(out,
                   "positive_volume_fraction",
                   concatenate(data.line_positive_volume_fractions,
                               data.polygon_positive_volume_fractions));
    writeSignedIndexArray(out,
                          "interface_marker",
                          concatenate(data.line_interface_markers,
                                      data.polygon_interface_markers));
    writeSignedIndexArray(out,
                          "parent_cell",
                          concatenate(data.line_parent_cells,
                                      data.polygon_parent_cells));
    out << "      </CellData>\n";
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">";
    for (const auto& point : data.points) {
        out << ' ' << point[0] << ' ' << point[1] << ' ' << point[2];
    }
    out << " </DataArray>\n";
    out << "      </Points>\n";
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
    out << "    \"geometry_tangent_policy\": ";
    writeJsonString(out, request.geometry_tangent_policy);
    out << ",\n";
    out << "    \"implicit_cut_root_tolerance\": "
        << request.implicit_cut_root_tolerance << ",\n";
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

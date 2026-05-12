/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Interfaces/LevelSetInterfaceGeometryWriter.h"

#include <array>
#include <iomanip>
#include <cstdint>
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
    std::vector<std::int64_t> line_interface_markers{};
    std::vector<std::int64_t> line_parent_cells{};
    std::vector<std::size_t> polygon_connectivity{};
    std::vector<std::size_t> polygon_offsets{};
    std::vector<std::array<Real, 3>> polygon_normals{};
    std::vector<Real> polygon_curvature_estimates{};
    std::vector<std::int64_t> polygon_interface_markers{};
    std::vector<std::int64_t> polygon_parent_cells{};
};

void appendFragmentCellData(const CutInterfaceFragment& fragment,
                            std::vector<std::array<Real, 3>>& normals,
                            std::vector<Real>& curvature_estimates,
                            std::vector<std::int64_t>& interface_markers,
                            std::vector<std::int64_t>& parent_cells)
{
    normals.push_back(fragment.normal);
    curvature_estimates.push_back(fragment.curvature_estimate);
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

} // namespace interfaces
} // namespace FE
} // namespace svmp

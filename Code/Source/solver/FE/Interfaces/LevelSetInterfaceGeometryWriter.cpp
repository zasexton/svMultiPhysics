/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Interfaces/LevelSetInterfaceGeometryWriter.h"

#include <array>
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
    std::vector<std::size_t> line_connectivity{};
    std::vector<std::size_t> line_offsets{};
    std::vector<std::size_t> polygon_connectivity{};
    std::vector<std::size_t> polygon_offsets{};
};

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
        }

        if (fragment.kind == CutInterfaceFragmentKind::Segment) {
            for (std::size_t i = 0; i < fragment.vertices.size(); ++i) {
                data.line_connectivity.push_back(first_point + i);
            }
            data.line_offsets.push_back(data.line_connectivity.size());
        } else {
            for (std::size_t i = 0; i < fragment.vertices.size(); ++i) {
                data.polygon_connectivity.push_back(first_point + i);
            }
            data.polygon_offsets.push_back(data.polygon_connectivity.size());
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

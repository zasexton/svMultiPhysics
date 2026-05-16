#include "Assembly/CutIntegrationContext.h"
#include "Assembly/CutDomainAssembler.h"
#include "Geometry/CutQuadrature.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "Systems/CutIntegrationInvalidation.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Search/CutCell.h"
#include "Mesh/Topology/CellTopology.h"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::assembly;
using namespace svmp::FE::geometry;
using namespace svmp::FE::interfaces;
using namespace svmp::FE::systems;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
namespace {

std::string join_cut_validity_messages(const std::vector<std::string>& messages)
{
    std::string joined;
    for (const auto& message : messages) {
        if (!joined.empty()) {
            joined += "; ";
        }
        joined += message;
    }
    return joined;
}

svmp::MeshBase make_cut_context_tetra_mesh()
{
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        std::vector<svmp::real_t>{
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0},
        std::vector<svmp::offset_t>{0, 4},
        std::vector<svmp::index_t>{0, 1, 2, 3},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Tetra, 4, 1}});
    mesh.finalize();
    return mesh;
}

svmp::search::EmbeddedGeometryDescriptor make_cut_context_plane()
{
    svmp::search::EmbeddedGeometryDescriptor embedded;
    embedded.kind = svmp::search::EmbeddedGeometryKind::Plane;
    embedded.origin = {{0.25, 0.0, 0.0}};
    embedded.normal = {{1.0, 0.0, 0.0}};
    embedded.geometry_epoch = 11;
    embedded.provenance.persistent_id = "context-plane";
    embedded.provenance.name = "context-plane";
    return embedded;
}

svmp::search::EmbeddedGeometryDescriptor make_axis_plane(int axis,
                                                         svmp::real_t coordinate,
                                                         std::uint64_t epoch,
                                                         const std::string& id)
{
    svmp::search::EmbeddedGeometryDescriptor embedded;
    embedded.kind = svmp::search::EmbeddedGeometryKind::Plane;
    embedded.origin = {{0.0, 0.0, 0.0}};
    embedded.origin[static_cast<std::size_t>(axis)] = coordinate;
    embedded.normal = {{0.0, 0.0, 0.0}};
    embedded.normal[static_cast<std::size_t>(axis)] = 1.0;
    embedded.geometry_epoch = epoch;
    embedded.provenance.persistent_id = id;
    embedded.provenance.name = id;
    return embedded;
}

std::array<svmp::real_t, 3> unit_vector_from_to(const std::array<svmp::real_t, 3>& from,
                                                const std::array<svmp::real_t, 3>& to)
{
    const std::array<svmp::real_t, 3> d{{to[0] - from[0], to[1] - from[1], to[2] - from[2]}};
    const auto n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (n <= svmp::real_t{0.0}) {
        return {{1.0, 0.0, 0.0}};
    }
    return {{d[0] / n, d[1] / n, d[2] / n}};
}

svmp::search::EmbeddedGeometryDescriptor make_context_sphere(
    std::array<svmp::real_t, 3> center,
    svmp::real_t radius,
    std::uint64_t epoch,
    const std::string& id)
{
    svmp::search::EmbeddedGeometryDescriptor embedded;
    embedded.kind = svmp::search::EmbeddedGeometryKind::Sphere;
    embedded.origin = center;
    embedded.radius = radius;
    embedded.geometry_epoch = epoch;
    embedded.revisions.geometry_epoch = epoch;
    embedded.provenance.persistent_id = id;
    embedded.provenance.name = id;
    embedded.provenance.provenance_epoch = epoch;
    return embedded;
}

svmp::search::EmbeddedGeometryDescriptor make_context_callback_sphere(
    std::array<svmp::real_t, 3> center,
    svmp::real_t radius,
    std::uint64_t epoch,
    const std::string& id)
{
    auto embedded = make_context_sphere(center, radius, epoch, id);
    embedded.kind = svmp::search::EmbeddedGeometryKind::SignedDistanceCallback;
    embedded.signed_distance_callback = [center, radius](const std::array<svmp::real_t, 3>& point) {
        const auto dx = point[0] - center[0];
        const auto dy = point[1] - center[1];
        const auto dz = point[2] - center[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz) - radius;
    };
    embedded.normal_callback = [center](const std::array<svmp::real_t, 3>& point) {
        return unit_vector_from_to(center, point);
    };
    embedded.closest_point_callback = [center, radius](const std::array<svmp::real_t, 3>& point) {
        const auto n = unit_vector_from_to(center, point);
        return std::array<svmp::real_t, 3>{{
            center[0] + radius * n[0],
            center[1] + radius * n[1],
            center[2] + radius * n[2]}};
    };
    return embedded;
}

svmp::search::EmbeddedGeometryDescriptor make_context_level_set_sphere(
    std::array<svmp::real_t, 3> center,
    svmp::real_t radius,
    std::uint64_t epoch,
    const std::string& id)
{
    svmp::search::EmbeddedGeometryDescriptor embedded;
    embedded.kind = svmp::search::EmbeddedGeometryKind::LevelSetField;
    embedded.origin = center;
    embedded.radius = radius;
    embedded.geometry_epoch = epoch;
    embedded.revisions.geometry_epoch = epoch;
    embedded.revisions.field_layout_revision = 1;
    embedded.revisions.field_value_revision = epoch;
    embedded.provenance.persistent_id = id;
    embedded.provenance.name = id;
    embedded.provenance.provenance_epoch = epoch;

    const std::array<std::array<svmp::real_t, 3>, 14> directions{{
        {{ 1.0,  0.0,  0.0}}, {{-1.0,  0.0,  0.0}},
        {{ 0.0,  1.0,  0.0}}, {{ 0.0, -1.0,  0.0}},
        {{ 0.0,  0.0,  1.0}}, {{ 0.0,  0.0, -1.0}},
        {{ 1.0,  1.0,  1.0}}, {{ 1.0,  1.0, -1.0}},
        {{ 1.0, -1.0,  1.0}}, {{ 1.0, -1.0, -1.0}},
        {{-1.0,  1.0,  1.0}}, {{-1.0,  1.0, -1.0}},
        {{-1.0, -1.0,  1.0}}, {{-1.0, -1.0, -1.0}}}};
    for (const auto& direction : directions) {
        const auto n = unit_vector_from_to({{0.0, 0.0, 0.0}}, direction);
        svmp::search::EmbeddedLevelSetSample sample;
        sample.point = {{center[0] + radius * n[0],
                         center[1] + radius * n[1],
                         center[2] + radius * n[2]}};
        sample.value = 0.0;
        sample.gradient = n;
        embedded.level_set_samples.push_back(sample);
    }
    return embedded;
}

svmp::search::EmbeddedGeometryDescriptor make_context_triangulated_octahedron_sphere(
    std::array<svmp::real_t, 3> center,
    svmp::real_t radius,
    std::uint64_t epoch,
    const std::string& id)
{
    svmp::search::EmbeddedGeometryDescriptor embedded;
    embedded.kind = svmp::search::EmbeddedGeometryKind::TriangulatedSurface;
    embedded.origin = center;
    embedded.radius = radius;
    embedded.geometry_epoch = epoch;
    embedded.revisions.geometry_epoch = epoch;
    embedded.revisions.source_surface_revision = epoch;
    embedded.provenance.persistent_id = id;
    embedded.provenance.name = id;
    embedded.provenance.provenance_epoch = epoch;

    const std::array<svmp::real_t, 3> xp{{center[0] + radius, center[1], center[2]}};
    const std::array<svmp::real_t, 3> xn{{center[0] - radius, center[1], center[2]}};
    const std::array<svmp::real_t, 3> yp{{center[0], center[1] + radius, center[2]}};
    const std::array<svmp::real_t, 3> yn{{center[0], center[1] - radius, center[2]}};
    const std::array<svmp::real_t, 3> zp{{center[0], center[1], center[2] + radius}};
    const std::array<svmp::real_t, 3> zn{{center[0], center[1], center[2] - radius}};

    const auto add_oriented_triangle =
        [&](std::array<svmp::real_t, 3> a,
            std::array<svmp::real_t, 3> b,
            std::array<svmp::real_t, 3> c) {
            const std::array<svmp::real_t, 3> ab{{b[0] - a[0], b[1] - a[1], b[2] - a[2]}};
            const std::array<svmp::real_t, 3> ac{{c[0] - a[0], c[1] - a[1], c[2] - a[2]}};
            const std::array<svmp::real_t, 3> normal{{
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0]}};
            const std::array<svmp::real_t, 3> centroid{{
                (a[0] + b[0] + c[0]) / 3.0,
                (a[1] + b[1] + c[1]) / 3.0,
                (a[2] + b[2] + c[2]) / 3.0}};
            const std::array<svmp::real_t, 3> outward{{
                centroid[0] - center[0],
                centroid[1] - center[1],
                centroid[2] - center[2]}};
            const auto orientation =
                normal[0] * outward[0] + normal[1] * outward[1] + normal[2] * outward[2];
            if (orientation < 0.0) {
                std::swap(b, c);
            }
            embedded.surface_triangles.push_back({{{a, b, c}}, embedded.provenance});
        };

    add_oriented_triangle(zp, xp, yp);
    add_oriented_triangle(zp, yp, xn);
    add_oriented_triangle(zp, xn, yn);
    add_oriented_triangle(zp, yn, xp);
    add_oriented_triangle(zn, yp, xp);
    add_oriented_triangle(zn, xn, yp);
    add_oriented_triangle(zn, yn, xn);
    add_oriented_triangle(zn, xp, yn);
    return embedded;
}

svmp::search::EmbeddedGeometryDescriptor make_context_boolean_composite(
    svmp::search::EmbeddedGeometryBooleanOperation operation,
    std::vector<svmp::search::EmbeddedGeometryDescriptor> children,
    std::uint64_t epoch,
    const std::string& id)
{
    svmp::search::EmbeddedGeometryDescriptor embedded;
    embedded.kind = svmp::search::EmbeddedGeometryKind::BooleanComposite;
    embedded.boolean_operation = operation;
    embedded.geometry_epoch = epoch;
    embedded.revisions.geometry_epoch = epoch;
    embedded.provenance.persistent_id = id;
    embedded.provenance.name = id;
    embedded.provenance.provenance_epoch = epoch;
    embedded.children = std::move(children);
    return embedded;
}

svmp::MeshBase make_cut_context_line_mesh()
{
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        std::vector<svmp::real_t>{0.0, 0.0, 0.0,
                                  1.0, 0.0, 0.0},
        std::vector<svmp::offset_t>{0, 2},
        std::vector<svmp::index_t>{0, 1},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Line, 2, 1}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_quadratic_line_mesh()
{
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        std::vector<svmp::real_t>{
            0.0, 0.00, 0.0,
            1.0, 0.00, 0.0,
            0.5, 0.08, 0.0},
        std::vector<svmp::offset_t>{0, 3},
        std::vector<svmp::index_t>{0, 1, 2},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Line, 2, 2}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_cubic_line_mesh()
{
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        std::vector<svmp::real_t>{
            0.00,  0.00, 0.0,
            0.22,  0.05, 0.0,
            0.68, -0.03, 0.0,
            1.05,  0.02, 0.0},
        std::vector<svmp::offset_t>{0, 4},
        std::vector<svmp::index_t>{0, 1, 2, 3},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Line, 2, 3}});
    mesh.finalize();
    return mesh;
}

std::vector<std::array<int, 4>> tetra_exponents_vtk_for_context_test(int p)
{
    std::vector<std::array<int, 4>> exps;
    exps.push_back({p, 0, 0, 0});
    exps.push_back({0, p, 0, 0});
    exps.push_back({0, 0, p, 0});
    exps.push_back({0, 0, 0, p});

    const auto eview = svmp::CellTopology::get_edges_view(svmp::CellFamily::Tetra);
    for (int ei = 0; ei < eview.edge_count; ++ei) {
        const int a = eview.pairs_flat[2 * ei + 0];
        const int b = eview.pairs_flat[2 * ei + 1];
        for (int k = 1; k <= p - 1; ++k) {
            std::array<int, 4> e{{0, 0, 0, 0}};
            e[static_cast<std::size_t>(a)] = p - k;
            e[static_cast<std::size_t>(b)] = k;
            exps.push_back(e);
        }
    }

    const auto fview = svmp::CellTopology::get_oriented_boundary_faces_view(svmp::CellFamily::Tetra);
    for (int fi = 0; fi < fview.face_count; ++fi) {
        const int b = fview.offsets[fi];
        const int e = fview.offsets[fi + 1];
        if (e - b != 3) {
            continue;
        }
        const int v0 = fview.indices[b + 0];
        const int v1 = fview.indices[b + 1];
        const int v2 = fview.indices[b + 2];
        for (int i = 1; i <= p - 2; ++i) {
            for (int j = 1; j <= p - 1 - i; ++j) {
                std::array<int, 4> f{{0, 0, 0, 0}};
                f[static_cast<std::size_t>(v0)] = p - i - j;
                f[static_cast<std::size_t>(v1)] = i;
                f[static_cast<std::size_t>(v2)] = j;
                exps.push_back(f);
            }
        }
    }

    for (int i = 1; i <= p - 3; ++i) {
        for (int j = 1; j <= p - 2 - i; ++j) {
            for (int k = 1; k <= p - 1 - i - j; ++k) {
                exps.push_back({p - i - j - k, i, j, k});
            }
        }
    }
    return exps;
}

svmp::MeshBase make_cut_context_cubic_tetra_mesh()
{
    constexpr int order = 3;
    const auto map_point = [](svmp::real_t r, svmp::real_t s, svmp::real_t t) {
        const svmp::real_t l0 = svmp::real_t{1.0} - r - s - t;
        return std::array<svmp::real_t, 3>{{
            r + svmp::real_t{0.04} * s * l0 + svmp::real_t{0.03} * t * l0,
            s + svmp::real_t{0.02} * r * l0,
            t + svmp::real_t{0.015} * r * s}};
    };

    const auto exps = tetra_exponents_vtk_for_context_test(order);
    std::vector<svmp::real_t> coords;
    coords.reserve(exps.size() * 3u);
    for (const auto& exp : exps) {
        const svmp::real_t r = static_cast<svmp::real_t>(exp[1]) / static_cast<svmp::real_t>(order);
        const svmp::real_t s = static_cast<svmp::real_t>(exp[2]) / static_cast<svmp::real_t>(order);
        const svmp::real_t t = static_cast<svmp::real_t>(exp[3]) / static_cast<svmp::real_t>(order);
        const auto point = map_point(r, s, t);
        coords.insert(coords.end(), point.begin(), point.end());
    }
    std::vector<svmp::index_t> connectivity(exps.size());
    for (std::size_t i = 0; i < connectivity.size(); ++i) {
        connectivity[i] = static_cast<svmp::index_t>(i);
    }

    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        coords,
        std::vector<svmp::offset_t>{0, static_cast<svmp::offset_t>(connectivity.size())},
        connectivity,
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Tetra, 4, order}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_triangle_mesh()
{
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        std::vector<svmp::real_t>{0.0, 0.0, 0.0,
                                  1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0},
        std::vector<svmp::offset_t>{0, 3},
        std::vector<svmp::index_t>{0, 1, 2},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Triangle, 3, 1}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_quadratic_tetra_mesh()
{
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        std::vector<svmp::real_t>{
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            0.50, -0.04, 0.0,
           -0.04, 0.50, 0.0,
            0.0, -0.03, 0.50,
            0.50, 0.50, 0.04,
            0.50, 0.0, 0.50,
            0.0, 0.50, 0.50},
        std::vector<svmp::offset_t>{0, 10},
        std::vector<svmp::index_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Tetra, 4, 2}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_quadratic_triangle_mesh()
{
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        std::vector<svmp::real_t>{
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.50, -0.10, 0.0,
            0.58, 0.58, 0.0,
           -0.10, 0.50, 0.0},
        std::vector<svmp::offset_t>{0, 6},
        std::vector<svmp::index_t>{0, 1, 2, 3, 4, 5},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Triangle, 3, 2}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_cubic_triangle_mesh()
{
    const auto p = [](svmp::real_t r, svmp::real_t s) {
        const svmp::real_t bubble = s * (svmp::real_t{1.0} - r - s);
        return std::array<svmp::real_t, 3>{{
            r + svmp::real_t{0.04} * bubble,
            s + svmp::real_t{0.02} * r * (svmp::real_t{1.0} - r - s),
            svmp::real_t{0.03} * r * s}};
    };
    const std::vector<std::array<svmp::real_t, 3>> pts{
        p(0.0, 0.0), p(1.0, 0.0), p(0.0, 1.0),
        p(1.0 / 3.0, 0.0), p(2.0 / 3.0, 0.0),
        p(2.0 / 3.0, 1.0 / 3.0), p(1.0 / 3.0, 2.0 / 3.0),
        p(0.0, 2.0 / 3.0), p(0.0, 1.0 / 3.0),
        p(1.0 / 3.0, 1.0 / 3.0)};
    std::vector<svmp::real_t> coords;
    coords.reserve(pts.size() * 3u);
    for (const auto& point : pts) {
        coords.insert(coords.end(), point.begin(), point.end());
    }
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        coords,
        std::vector<svmp::offset_t>{0, 10},
        std::vector<svmp::index_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Triangle, 3, 3}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_cubic_non_graph_triangle_mesh()
{
    const auto p = [](svmp::real_t r, svmp::real_t s) {
        const svmp::real_t centered = r - svmp::real_t{0.5};
        return std::array<svmp::real_t, 3>{{
            centered * centered,
            s,
            svmp::real_t{0.02} * r * s}};
    };
    const std::vector<std::array<svmp::real_t, 3>> pts{
        p(0.0, 0.0), p(1.0, 0.0), p(0.0, 1.0),
        p(1.0 / 3.0, 0.0), p(2.0 / 3.0, 0.0),
        p(2.0 / 3.0, 1.0 / 3.0), p(1.0 / 3.0, 2.0 / 3.0),
        p(0.0, 2.0 / 3.0), p(0.0, 1.0 / 3.0),
        p(1.0 / 3.0, 1.0 / 3.0)};
    std::vector<svmp::real_t> coords;
    coords.reserve(pts.size() * 3u);
    for (const auto& point : pts) {
        coords.insert(coords.end(), point.begin(), point.end());
    }
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        coords,
        std::vector<svmp::offset_t>{0, 10},
        std::vector<svmp::index_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Triangle, 3, 3}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_quadratic_quad_mesh()
{
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        std::vector<svmp::real_t>{
           -1.0, -1.0, 0.0,
            1.0, -1.0, 0.0,
            1.0,  1.0, 0.0,
           -1.0,  1.0, 0.0,
            0.0, -1.0, 0.12,
            1.0,  0.0, 0.08,
            0.0,  1.0, 0.10,
           -1.0,  0.0, -0.06},
        std::vector<svmp::offset_t>{0, 8},
        std::vector<svmp::index_t>{0, 1, 2, 3, 4, 5, 6, 7},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Quad, 4, 2}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_cubic_quad_mesh()
{
    const auto p = [](int i, int j) {
        const svmp::real_t r =
            svmp::real_t{-1.0} + svmp::real_t{2.0} * static_cast<svmp::real_t>(i) / svmp::real_t{3.0};
        const svmp::real_t s =
            svmp::real_t{-1.0} + svmp::real_t{2.0} * static_cast<svmp::real_t>(j) / svmp::real_t{3.0};
        return std::array<svmp::real_t, 3>{{
            svmp::real_t{0.5} * (r + svmp::real_t{1.0}) +
                svmp::real_t{0.06} * s * (svmp::real_t{1.0} - r * r),
            s,
            svmp::real_t{0.04} * r * s}};
    };
    const std::vector<std::array<svmp::real_t, 3>> pts{
        p(0, 0), p(3, 0), p(3, 3), p(0, 3),
        p(1, 0), p(2, 0),
        p(3, 1), p(3, 2),
        p(2, 3), p(1, 3),
        p(0, 2), p(0, 1),
        p(1, 1), p(1, 2), p(2, 1), p(2, 2)};
    std::vector<svmp::real_t> coords;
    coords.reserve(pts.size() * 3u);
    for (const auto& point : pts) {
        coords.insert(coords.end(), point.begin(), point.end());
    }
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        coords,
        std::vector<svmp::offset_t>{0, 16},
        std::vector<svmp::index_t>{0, 1, 2, 3, 4, 5, 6, 7,
                                   8, 9, 10, 11, 12, 13, 14, 15},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Quad, 4, 3}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_cubic_non_graph_quad_mesh()
{
    const auto p = [](int i, int j) {
        const svmp::real_t r =
            svmp::real_t{-1.0} + svmp::real_t{2.0} * static_cast<svmp::real_t>(i) / svmp::real_t{3.0};
        const svmp::real_t s =
            svmp::real_t{-1.0} + svmp::real_t{2.0} * static_cast<svmp::real_t>(j) / svmp::real_t{3.0};
        return std::array<svmp::real_t, 3>{{
            r * r,
            s,
            svmp::real_t{0.02} * r * s}};
    };
    const std::vector<std::array<svmp::real_t, 3>> pts{
        p(0, 0), p(3, 0), p(3, 3), p(0, 3),
        p(1, 0), p(2, 0),
        p(3, 1), p(3, 2),
        p(2, 3), p(1, 3),
        p(0, 2), p(0, 1),
        p(1, 1), p(1, 2), p(2, 1), p(2, 2)};
    std::vector<svmp::real_t> coords;
    coords.reserve(pts.size() * 3u);
    for (const auto& point : pts) {
        coords.insert(coords.end(), point.begin(), point.end());
    }
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        coords,
        std::vector<svmp::offset_t>{0, 16},
        std::vector<svmp::index_t>{0, 1, 2, 3, 4, 5, 6, 7,
                                   8, 9, 10, 11, 12, 13, 14, 15},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Quad, 4, 3}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_quadratic_hex_mesh()
{
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        std::vector<svmp::real_t>{
           -1.0, -1.0, -1.0,
            1.0, -1.0, -1.0,
            1.0,  1.0, -1.0,
           -1.0,  1.0, -1.0,
           -1.0, -1.0,  1.0,
            1.0, -1.0,  1.0,
            1.0,  1.0,  1.0,
           -1.0,  1.0,  1.0,
            0.0, -1.0, -0.88,
            1.0,  0.0, -0.92,
            0.0,  1.0, -0.90,
           -1.0,  0.0, -1.08,
            0.0, -1.0,  1.06,
            1.0,  0.0,  0.94,
            0.0,  1.0,  1.04,
           -1.0,  0.0,  0.96,
           -1.08, -1.0,  0.0,
            1.06, -1.0,  0.0,
            1.04,  1.0,  0.0,
           -0.96,  1.0,  0.0},
        std::vector<svmp::offset_t>{0, 20},
        std::vector<svmp::index_t>{0, 1, 2, 3, 4, 5, 6, 7,
                                   8, 9, 10, 11, 12, 13, 14, 15,
                                   16, 17, 18, 19},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Hex, 8, 2}});
    mesh.finalize();
    return mesh;
}

std::vector<std::array<int, 3>> hex_lagrange_indices_vtk_for_context_test(int p)
{
    std::vector<std::array<int, 3>> idx;
    const std::array<std::array<int, 3>, 8> corner_grid{{
        {{0, 0, 0}}, {{p, 0, 0}}, {{p, p, 0}}, {{0, p, 0}},
        {{0, 0, p}}, {{p, 0, p}}, {{p, p, p}}, {{0, p, p}}}};
    for (const auto& corner : corner_grid) {
        idx.push_back(corner);
    }

    const auto eview = svmp::CellTopology::get_edges_view(svmp::CellFamily::Hex);
    for (int ei = 0; ei < eview.edge_count; ++ei) {
        const int a = eview.pairs_flat[2 * ei + 0];
        const int b = eview.pairs_flat[2 * ei + 1];
        const auto A = corner_grid[static_cast<std::size_t>(a)];
        const auto B = corner_grid[static_cast<std::size_t>(b)];
        for (int k = 1; k <= p - 1; ++k) {
            const svmp::real_t u = static_cast<svmp::real_t>(k) / static_cast<svmp::real_t>(p);
            idx.push_back({{
                static_cast<int>(std::lround((svmp::real_t{1.0} - u) * A[0] + u * B[0])),
                static_cast<int>(std::lround((svmp::real_t{1.0} - u) * A[1] + u * B[1])),
                static_cast<int>(std::lround((svmp::real_t{1.0} - u) * A[2] + u * B[2]))}});
        }
    }

    const auto fview = svmp::CellTopology::get_oriented_boundary_faces_view(svmp::CellFamily::Hex);
    for (int fi = 0; fi < fview.face_count; ++fi) {
        const int b = fview.offsets[fi];
        const int e = fview.offsets[fi + 1];
        if (e - b != 4) {
            continue;
        }
        const auto A = corner_grid[static_cast<std::size_t>(fview.indices[b + 0])];
        const auto B = corner_grid[static_cast<std::size_t>(fview.indices[b + 1])];
        const auto C = corner_grid[static_cast<std::size_t>(fview.indices[b + 2])];
        const auto D = corner_grid[static_cast<std::size_t>(fview.indices[b + 3])];
        for (int i = 1; i <= p - 1; ++i) {
            for (int j = 1; j <= p - 1; ++j) {
                const svmp::real_t u = static_cast<svmp::real_t>(i) / static_cast<svmp::real_t>(p);
                const svmp::real_t v = static_cast<svmp::real_t>(j) / static_cast<svmp::real_t>(p);
                std::array<int, 3> g{{0, 0, 0}};
                for (int d = 0; d < 3; ++d) {
                    const auto dim = static_cast<std::size_t>(d);
                    const svmp::real_t x =
                        (svmp::real_t{1.0} - u) * (svmp::real_t{1.0} - v) * A[dim] +
                        u * (svmp::real_t{1.0} - v) * B[dim] +
                        u * v * C[dim] +
                        (svmp::real_t{1.0} - u) * v * D[dim];
                    g[dim] = static_cast<int>(std::lround(x));
                }
                idx.push_back(g);
            }
        }
    }

    for (int i = 1; i <= p - 1; ++i) {
        for (int j = 1; j <= p - 1; ++j) {
            for (int k = 1; k <= p - 1; ++k) {
                idx.push_back({{i, j, k}});
            }
        }
    }
    return idx;
}

svmp::MeshBase make_cut_context_cubic_hex_mesh()
{
    constexpr int order = 3;
    const auto map_point = [](svmp::real_t r, svmp::real_t s, svmp::real_t t) {
        return std::array<svmp::real_t, 3>{{
            svmp::real_t{0.5} * (r + svmp::real_t{1.0}) +
                svmp::real_t{0.04} * s * (svmp::real_t{1.0} - r * r) +
                svmp::real_t{0.02} * t * (svmp::real_t{1.0} - r * r),
            s + svmp::real_t{0.03} * r * (svmp::real_t{1.0} - s * s),
            t + svmp::real_t{0.02} * r * s * (svmp::real_t{1.0} - t * t)}};
    };

    const auto labels = hex_lagrange_indices_vtk_for_context_test(order);
    std::vector<svmp::real_t> coords;
    coords.reserve(labels.size() * 3u);
    for (const auto& label : labels) {
        const svmp::real_t r = svmp::real_t{-1.0} +
            svmp::real_t{2.0} * static_cast<svmp::real_t>(label[0]) /
            static_cast<svmp::real_t>(order);
        const svmp::real_t s = svmp::real_t{-1.0} +
            svmp::real_t{2.0} * static_cast<svmp::real_t>(label[1]) /
            static_cast<svmp::real_t>(order);
        const svmp::real_t t = svmp::real_t{-1.0} +
            svmp::real_t{2.0} * static_cast<svmp::real_t>(label[2]) /
            static_cast<svmp::real_t>(order);
        const auto point = map_point(r, s, t);
        coords.insert(coords.end(), point.begin(), point.end());
    }
    std::vector<svmp::index_t> connectivity(labels.size());
    for (std::size_t i = 0; i < connectivity.size(); ++i) {
        connectivity[i] = static_cast<svmp::index_t>(i);
    }

    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        coords,
        std::vector<svmp::offset_t>{0, static_cast<svmp::offset_t>(connectivity.size())},
        connectivity,
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Hex, 8, order}});
    mesh.finalize();
    return mesh;
}

struct WedgeLagrangeLabelForContextTest {
    std::array<int, 3> tri_exp{{0, 0, 0}};
    int kz{0};
};

std::array<svmp::real_t, 3> wedge_corner_parametric_for_context_test(int local_corner)
{
    switch (local_corner) {
        case 0: return {{0.0, 0.0, -1.0}};
        case 1: return {{1.0, 0.0, -1.0}};
        case 2: return {{0.0, 1.0, -1.0}};
        case 3: return {{0.0, 0.0,  1.0}};
        case 4: return {{1.0, 0.0,  1.0}};
        case 5: return {{0.0, 1.0,  1.0}};
        default: return {{0.0, 0.0, 0.0}};
    }
}

WedgeLagrangeLabelForContextTest wedge_label_from_param_for_context_test(
    int p,
    const std::array<svmp::real_t, 3>& xi)
{
    int a1 = static_cast<int>(std::lround(static_cast<svmp::real_t>(p) * xi[0]));
    int a2 = static_cast<int>(std::lround(static_cast<svmp::real_t>(p) * xi[1]));
    a1 = std::clamp(a1, 0, p);
    a2 = std::clamp(a2, 0, p);
    int a0 = p - a1 - a2;
    if (a0 < 0) {
        a0 = 0;
    }
    int kz = static_cast<int>(std::lround((xi[2] + svmp::real_t{1.0}) *
                                          svmp::real_t{0.5} *
                                          static_cast<svmp::real_t>(p)));
    kz = std::clamp(kz, 0, p);
    return {{{a0, a1, a2}}, kz};
}

std::vector<WedgeLagrangeLabelForContextTest> wedge_lagrange_labels_vtk_for_context_test(int p)
{
    std::vector<WedgeLagrangeLabelForContextTest> labels;
    labels.reserve(static_cast<std::size_t>((p + 1) * (p + 1) * (p + 2) / 2));

    for (int c = 0; c < 6; ++c) {
        labels.push_back(wedge_label_from_param_for_context_test(
            p, wedge_corner_parametric_for_context_test(c)));
    }

    const auto eview = svmp::CellTopology::get_edges_view(svmp::CellFamily::Wedge);
    for (int ei = 0; ei < eview.edge_count; ++ei) {
        const int a = eview.pairs_flat[2 * ei + 0];
        const int b = eview.pairs_flat[2 * ei + 1];
        const auto A = wedge_corner_parametric_for_context_test(a);
        const auto B = wedge_corner_parametric_for_context_test(b);
        for (int k = 1; k <= p - 1; ++k) {
            const svmp::real_t u = static_cast<svmp::real_t>(k) / static_cast<svmp::real_t>(p);
            labels.push_back(wedge_label_from_param_for_context_test(
                p,
                {{(svmp::real_t{1.0} - u) * A[0] + u * B[0],
                  (svmp::real_t{1.0} - u) * A[1] + u * B[1],
                  (svmp::real_t{1.0} - u) * A[2] + u * B[2]}}));
        }
    }

    const auto fview = svmp::CellTopology::get_oriented_boundary_faces_view(svmp::CellFamily::Wedge);
    for (int fi = 0; fi < fview.face_count; ++fi) {
        const int b = fview.offsets[fi];
        const int e = fview.offsets[fi + 1];
        const int fv = e - b;
        if (fv == 3) {
            const auto A = wedge_corner_parametric_for_context_test(fview.indices[b + 0]);
            const auto B = wedge_corner_parametric_for_context_test(fview.indices[b + 1]);
            const auto C = wedge_corner_parametric_for_context_test(fview.indices[b + 2]);
            for (int i = 1; i <= p - 2; ++i) {
                for (int j = 1; j <= p - 1 - i; ++j) {
                    const svmp::real_t w1 = static_cast<svmp::real_t>(i) / static_cast<svmp::real_t>(p);
                    const svmp::real_t w2 = static_cast<svmp::real_t>(j) / static_cast<svmp::real_t>(p);
                    const svmp::real_t w0 = svmp::real_t{1.0} - w1 - w2;
                    labels.push_back(wedge_label_from_param_for_context_test(
                        p,
                        {{w0 * A[0] + w1 * B[0] + w2 * C[0],
                          w0 * A[1] + w1 * B[1] + w2 * C[1],
                          w0 * A[2] + w1 * B[2] + w2 * C[2]}}));
                }
            }
        } else if (fv == 4) {
            const auto A = wedge_corner_parametric_for_context_test(fview.indices[b + 0]);
            const auto B = wedge_corner_parametric_for_context_test(fview.indices[b + 1]);
            const auto C = wedge_corner_parametric_for_context_test(fview.indices[b + 2]);
            const auto D = wedge_corner_parametric_for_context_test(fview.indices[b + 3]);
            for (int i = 1; i <= p - 1; ++i) {
                for (int j = 1; j <= p - 1; ++j) {
                    const svmp::real_t u = static_cast<svmp::real_t>(i) / static_cast<svmp::real_t>(p);
                    const svmp::real_t v = static_cast<svmp::real_t>(j) / static_cast<svmp::real_t>(p);
                    labels.push_back(wedge_label_from_param_for_context_test(
                        p,
                        {{(svmp::real_t{1.0} - u) * (svmp::real_t{1.0} - v) * A[0] +
                              u * (svmp::real_t{1.0} - v) * B[0] + u * v * C[0] +
                              (svmp::real_t{1.0} - u) * v * D[0],
                          (svmp::real_t{1.0} - u) * (svmp::real_t{1.0} - v) * A[1] +
                              u * (svmp::real_t{1.0} - v) * B[1] + u * v * C[1] +
                              (svmp::real_t{1.0} - u) * v * D[1],
                          (svmp::real_t{1.0} - u) * (svmp::real_t{1.0} - v) * A[2] +
                              u * (svmp::real_t{1.0} - v) * B[2] + u * v * C[2] +
                              (svmp::real_t{1.0} - u) * v * D[2]}}));
                }
            }
        }
    }

    for (int i = 1; i <= p - 2; ++i) {
        for (int j = 1; j <= p - 1 - i; ++j) {
            for (int k = 1; k <= p - 1; ++k) {
                labels.push_back({{{p - i - j, i, j}}, k});
            }
        }
    }
    return labels;
}

svmp::MeshBase make_cut_context_cubic_wedge_mesh()
{
    constexpr int order = 3;
    const auto map_point = [](svmp::real_t xi, svmp::real_t eta, svmp::real_t zeta) {
        const svmp::real_t l0 = svmp::real_t{1.0} - xi - eta;
        return std::array<svmp::real_t, 3>{{
            svmp::real_t{0.5} * (zeta + svmp::real_t{1.0}) +
                svmp::real_t{0.04} * xi * (svmp::real_t{1.0} - zeta * zeta) +
                svmp::real_t{0.02} * eta * (svmp::real_t{1.0} - zeta * zeta),
            xi + svmp::real_t{0.02} * zeta * xi * l0,
            eta + svmp::real_t{0.015} * zeta * xi * eta}};
    };

    const auto labels = wedge_lagrange_labels_vtk_for_context_test(order);
    std::vector<svmp::real_t> coords;
    coords.reserve(labels.size() * 3u);
    for (const auto& label : labels) {
        const svmp::real_t xi = static_cast<svmp::real_t>(label.tri_exp[1]) /
                                static_cast<svmp::real_t>(order);
        const svmp::real_t eta = static_cast<svmp::real_t>(label.tri_exp[2]) /
                                 static_cast<svmp::real_t>(order);
        const svmp::real_t zeta = svmp::real_t{-1.0} +
            svmp::real_t{2.0} * static_cast<svmp::real_t>(label.kz) /
            static_cast<svmp::real_t>(order);
        const auto point = map_point(xi, eta, zeta);
        coords.insert(coords.end(), point.begin(), point.end());
    }
    std::vector<svmp::index_t> connectivity(labels.size());
    for (std::size_t i = 0; i < connectivity.size(); ++i) {
        connectivity[i] = static_cast<svmp::index_t>(i);
    }

    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        coords,
        std::vector<svmp::offset_t>{0, static_cast<svmp::offset_t>(connectivity.size())},
        connectivity,
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Wedge, 6, order}});
    mesh.finalize();
    return mesh;
}

std::array<svmp::real_t, 3> pyramid_corner_parametric_for_context_test(int local_corner)
{
    switch (local_corner) {
        case 0: return {{-1.0, -1.0, 0.0}};
        case 1: return {{ 1.0, -1.0, 0.0}};
        case 2: return {{ 1.0,  1.0, 0.0}};
        case 3: return {{-1.0,  1.0, 0.0}};
        case 4: return {{ 0.0,  0.0, 1.0}};
        default: return {{0.0, 0.0, 0.0}};
    }
}

std::array<svmp::real_t, 3> interpolate_pyramid_parametric_for_context_test(
    const std::array<svmp::real_t, 3>& a,
    const std::array<svmp::real_t, 3>& b,
    svmp::real_t u)
{
    return {{(svmp::real_t{1.0} - u) * a[0] + u * b[0],
             (svmp::real_t{1.0} - u) * a[1] + u * b[1],
             (svmp::real_t{1.0} - u) * a[2] + u * b[2]}};
}

std::array<svmp::real_t, 3> bilerp_pyramid_parametric_for_context_test(
    const std::array<svmp::real_t, 3>& a,
    const std::array<svmp::real_t, 3>& b,
    const std::array<svmp::real_t, 3>& c,
    const std::array<svmp::real_t, 3>& d,
    svmp::real_t u,
    svmp::real_t v)
{
    const svmp::real_t w00 = (svmp::real_t{1.0} - u) * (svmp::real_t{1.0} - v);
    const svmp::real_t w10 = u * (svmp::real_t{1.0} - v);
    const svmp::real_t w11 = u * v;
    const svmp::real_t w01 = (svmp::real_t{1.0} - u) * v;
    return {{w00 * a[0] + w10 * b[0] + w11 * c[0] + w01 * d[0],
             w00 * a[1] + w10 * b[1] + w11 * c[1] + w01 * d[1],
             w00 * a[2] + w10 * b[2] + w11 * c[2] + w01 * d[2]}};
}

std::vector<std::array<svmp::real_t, 3>> pyramid_lagrange_parametric_points_vtk_for_context_test(int p)
{
    const auto pattern =
        svmp::CellTopology::high_order_pattern(
            svmp::CellFamily::Pyramid, p, svmp::CellTopology::HighOrderKind::Lagrange);
    const auto eview = svmp::CellTopology::get_edges_view(svmp::CellFamily::Pyramid);
    const auto fview = svmp::CellTopology::get_oriented_boundary_faces_view(svmp::CellFamily::Pyramid);

    std::vector<std::array<svmp::real_t, 3>> points;
    points.reserve(pattern.sequence.size());
    for (const auto& role : pattern.sequence) {
        switch (role.role) {
            case svmp::CellTopology::HONodeRole::Corner:
                points.push_back(pyramid_corner_parametric_for_context_test(role.idx0));
                break;
            case svmp::CellTopology::HONodeRole::Edge: {
                const int edge = role.idx0;
                const int a = eview.pairs_flat[2 * edge + 0];
                const int b = eview.pairs_flat[2 * edge + 1];
                const svmp::real_t u =
                    static_cast<svmp::real_t>(role.idx1) / static_cast<svmp::real_t>(p);
                points.push_back(interpolate_pyramid_parametric_for_context_test(
                    pyramid_corner_parametric_for_context_test(a),
                    pyramid_corner_parametric_for_context_test(b),
                    u));
                break;
            }
            case svmp::CellTopology::HONodeRole::Face: {
                const int face = role.idx0;
                const int begin = fview.offsets[face];
                const int end = fview.offsets[face + 1];
                const int vertices = end - begin;
                if (vertices == 3) {
                    const auto a = pyramid_corner_parametric_for_context_test(fview.indices[begin + 0]);
                    const auto b = pyramid_corner_parametric_for_context_test(fview.indices[begin + 1]);
                    const auto c = pyramid_corner_parametric_for_context_test(fview.indices[begin + 2]);
                    const svmp::real_t w1 =
                        static_cast<svmp::real_t>(role.idx1) / static_cast<svmp::real_t>(p);
                    const svmp::real_t w2 =
                        static_cast<svmp::real_t>(role.idx2) / static_cast<svmp::real_t>(p);
                    const svmp::real_t w0 = svmp::real_t{1.0} - w1 - w2;
                    points.push_back({{w0 * a[0] + w1 * b[0] + w2 * c[0],
                                       w0 * a[1] + w1 * b[1] + w2 * c[1],
                                       w0 * a[2] + w1 * b[2] + w2 * c[2]}});
                } else if (vertices == 4) {
                    const svmp::real_t u =
                        static_cast<svmp::real_t>(role.idx1) / static_cast<svmp::real_t>(p);
                    const svmp::real_t v =
                        static_cast<svmp::real_t>(role.idx2) / static_cast<svmp::real_t>(p);
                    points.push_back(bilerp_pyramid_parametric_for_context_test(
                        pyramid_corner_parametric_for_context_test(fview.indices[begin + 0]),
                        pyramid_corner_parametric_for_context_test(fview.indices[begin + 1]),
                        pyramid_corner_parametric_for_context_test(fview.indices[begin + 2]),
                        pyramid_corner_parametric_for_context_test(fview.indices[begin + 3]),
                        u,
                        v));
                }
                break;
            }
            case svmp::CellTopology::HONodeRole::Volume: {
                const svmp::real_t zeta =
                    static_cast<svmp::real_t>(role.idx2) / static_cast<svmp::real_t>(p);
                const svmp::real_t scale = svmp::real_t{1.0} - zeta;
                const int layer_nodes = (p + 1) - role.idx2;
                const int denominator = layer_nodes - 1;
                const svmp::real_t u = svmp::real_t{-1.0} +
                    svmp::real_t{2.0} * static_cast<svmp::real_t>(role.idx0) /
                        static_cast<svmp::real_t>(denominator);
                const svmp::real_t v = svmp::real_t{-1.0} +
                    svmp::real_t{2.0} * static_cast<svmp::real_t>(role.idx1) /
                        static_cast<svmp::real_t>(denominator);
                points.push_back({{scale * u, scale * v, zeta}});
                break;
            }
        }
    }
    return points;
}

svmp::MeshBase make_cut_context_cubic_pyramid_mesh()
{
    constexpr int order = 3;
    const auto map_point = [](svmp::real_t xi, svmp::real_t eta, svmp::real_t zeta) {
        return std::array<svmp::real_t, 3>{{
            zeta + svmp::real_t{0.035} * xi * (svmp::real_t{1.0} - zeta) +
                svmp::real_t{0.02} * eta * (svmp::real_t{1.0} - zeta),
            xi + svmp::real_t{0.02} * zeta * xi * (svmp::real_t{1.0} - zeta),
            eta + svmp::real_t{0.015} * zeta * xi * eta}};
    };

    const auto points = pyramid_lagrange_parametric_points_vtk_for_context_test(order);
    std::vector<svmp::real_t> coords;
    coords.reserve(points.size() * 3u);
    for (const auto& xi : points) {
        const auto point = map_point(xi[0], xi[1], xi[2]);
        coords.insert(coords.end(), point.begin(), point.end());
    }
    std::vector<svmp::index_t> connectivity(points.size());
    for (std::size_t i = 0; i < connectivity.size(); ++i) {
        connectivity[i] = static_cast<svmp::index_t>(i);
    }

    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        coords,
        std::vector<svmp::offset_t>{0, static_cast<svmp::offset_t>(connectivity.size())},
        connectivity,
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Pyramid, 5, order}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_cubic_non_graph_hex_mesh()
{
    constexpr int order = 3;
    const auto labels = hex_lagrange_indices_vtk_for_context_test(order);
    std::vector<svmp::real_t> coords;
    coords.reserve(labels.size() * 3u);
    for (const auto& label : labels) {
        const svmp::real_t r = svmp::real_t{-1.0} +
            svmp::real_t{2.0} * static_cast<svmp::real_t>(label[0]) /
            static_cast<svmp::real_t>(order);
        const svmp::real_t s = svmp::real_t{-1.0} +
            svmp::real_t{2.0} * static_cast<svmp::real_t>(label[1]) /
            static_cast<svmp::real_t>(order);
        const svmp::real_t t = svmp::real_t{-1.0} +
            svmp::real_t{2.0} * static_cast<svmp::real_t>(label[2]) /
            static_cast<svmp::real_t>(order);
        const std::array<svmp::real_t, 3> point{{r * r, s, t}};
        coords.insert(coords.end(), point.begin(), point.end());
    }
    std::vector<svmp::index_t> connectivity(labels.size());
    for (std::size_t i = 0; i < connectivity.size(); ++i) {
        connectivity[i] = static_cast<svmp::index_t>(i);
    }

    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        coords,
        std::vector<svmp::offset_t>{0, static_cast<svmp::offset_t>(connectivity.size())},
        connectivity,
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Hex, 8, order}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_cubic_non_graph_wedge_mesh()
{
    constexpr int order = 3;
    const auto labels = wedge_lagrange_labels_vtk_for_context_test(order);
    std::vector<svmp::real_t> coords;
    coords.reserve(labels.size() * 3u);
    for (const auto& label : labels) {
        const svmp::real_t xi = static_cast<svmp::real_t>(label.tri_exp[1]) /
                                static_cast<svmp::real_t>(order);
        const svmp::real_t eta = static_cast<svmp::real_t>(label.tri_exp[2]) /
                                 static_cast<svmp::real_t>(order);
        const svmp::real_t zeta = svmp::real_t{-1.0} +
            svmp::real_t{2.0} * static_cast<svmp::real_t>(label.kz) /
            static_cast<svmp::real_t>(order);
        const std::array<svmp::real_t, 3> point{{zeta * zeta, xi, eta}};
        coords.insert(coords.end(), point.begin(), point.end());
    }
    std::vector<svmp::index_t> connectivity(labels.size());
    for (std::size_t i = 0; i < connectivity.size(); ++i) {
        connectivity[i] = static_cast<svmp::index_t>(i);
    }

    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        coords,
        std::vector<svmp::offset_t>{0, static_cast<svmp::offset_t>(connectivity.size())},
        connectivity,
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Wedge, 6, order}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_cubic_non_graph_pyramid_mesh()
{
    constexpr int order = 3;
    const auto points = pyramid_lagrange_parametric_points_vtk_for_context_test(order);
    std::vector<svmp::real_t> coords;
    coords.reserve(points.size() * 3u);
    for (const auto& xi : points) {
        const svmp::real_t centered = xi[2] - svmp::real_t{0.5};
        const std::array<svmp::real_t, 3> point{{centered * centered, xi[0], xi[1]}};
        coords.insert(coords.end(), point.begin(), point.end());
    }
    std::vector<svmp::index_t> connectivity(points.size());
    for (std::size_t i = 0; i < connectivity.size(); ++i) {
        connectivity[i] = static_cast<svmp::index_t>(i);
    }

    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        coords,
        std::vector<svmp::offset_t>{0, static_cast<svmp::offset_t>(connectivity.size())},
        connectivity,
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Pyramid, 5, order}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_quad_mesh()
{
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        std::vector<svmp::real_t>{0.0, 0.0, 0.0,
                                  1.0, 0.0, 0.0,
                                  1.0, 1.0, 0.0,
                                  0.0, 1.0, 0.0},
        std::vector<svmp::offset_t>{0, 4},
        std::vector<svmp::index_t>{0, 1, 2, 3},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Quad, 4, 1}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_hex_mesh()
{
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        std::vector<svmp::real_t>{0.0, 0.0, 0.0,
                                  1.0, 0.0, 0.0,
                                  1.0, 1.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0,
                                  1.0, 0.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  0.0, 1.0, 1.0},
        std::vector<svmp::offset_t>{0, 8},
        std::vector<svmp::index_t>{0, 1, 2, 3, 4, 5, 6, 7},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Hex, 8, 1}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_wedge_mesh()
{
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        std::vector<svmp::real_t>{0.0, 0.0, 0.0,
                                  1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0,
                                  1.0, 0.0, 1.0,
                                  0.0, 1.0, 1.0},
        std::vector<svmp::offset_t>{0, 6},
        std::vector<svmp::index_t>{0, 1, 2, 3, 4, 5},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Wedge, 6, 1}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_pyramid_mesh()
{
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        std::vector<svmp::real_t>{0.0, 0.0, 0.0,
                                  1.0, 0.0, 0.0,
                                  1.0, 1.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.5, 0.5, 1.0},
        std::vector<svmp::offset_t>{0, 5},
        std::vector<svmp::index_t>{0, 1, 2, 3, 4},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Pyramid, 5, 1}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_polygon_mesh()
{
    svmp::MeshBase mesh;
    mesh.build_from_arrays(
        3,
        std::vector<svmp::real_t>{0.0, 0.0, 0.0,
                                  1.0, 0.0, 0.0,
                                  1.2, 0.6, 0.0,
                                  0.5, 1.2, 0.0,
                                 -0.2, 0.6, 0.0},
        std::vector<svmp::offset_t>{0, 5},
        std::vector<svmp::index_t>{0, 1, 2, 3, 4},
        std::vector<svmp::CellShape>{svmp::CellShape{svmp::CellFamily::Polygon, 5, 1}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_cut_context_polyhedron_cube_mesh()
{
    auto mesh = make_cut_context_hex_mesh();
    const std::vector<svmp::CellShape> face_shapes(6, svmp::CellShape{svmp::CellFamily::Quad, 4, 1});
    mesh.set_faces_from_arrays(
        face_shapes,
        std::vector<svmp::offset_t>{0, 4, 8, 12, 16, 20, 24},
        std::vector<svmp::index_t>{0, 1, 2, 3,
                                   4, 5, 6, 7,
                                   0, 1, 5, 4,
                                   1, 2, 6, 5,
                                   2, 3, 7, 6,
                                   3, 0, 4, 7},
        std::vector<std::array<svmp::index_t, 2>>{
            {{0, svmp::INVALID_INDEX}}, {{0, svmp::INVALID_INDEX}}, {{0, svmp::INVALID_INDEX}},
            {{0, svmp::INVALID_INDEX}}, {{0, svmp::INVALID_INDEX}}, {{0, svmp::INVALID_INDEX}}});
    mesh.finalize();
    return mesh;
}

svmp::MeshBase make_first_order_cut_context_mesh(svmp::CellFamily family)
{
    switch (family) {
        case svmp::CellFamily::Line:
            return make_cut_context_line_mesh();
        case svmp::CellFamily::Triangle:
            return make_cut_context_triangle_mesh();
        case svmp::CellFamily::Quad:
            return make_cut_context_quad_mesh();
        case svmp::CellFamily::Tetra:
            return make_cut_context_tetra_mesh();
        case svmp::CellFamily::Hex:
            return make_cut_context_hex_mesh();
        case svmp::CellFamily::Wedge:
            return make_cut_context_wedge_mesh();
        case svmp::CellFamily::Pyramid:
            return make_cut_context_pyramid_mesh();
        case svmp::CellFamily::Polygon:
            return make_cut_context_polygon_mesh();
        case svmp::CellFamily::Polyhedron:
            return make_cut_context_polyhedron_cube_mesh();
        default:
            return make_cut_context_tetra_mesh();
    }
}

svmp::search::EmbeddedGeometryDescriptor make_first_order_linearized_context_embedded(
    svmp::search::EmbeddedGeometryKind kind,
    std::uint64_t epoch,
    const std::string& id)
{
    constexpr std::array<svmp::real_t, 3> center{{0.0, 0.0, 0.0}};
    constexpr svmp::real_t radius = 0.55;
    switch (kind) {
        case svmp::search::EmbeddedGeometryKind::Sphere:
            return make_context_sphere(center, radius, epoch, id);
        case svmp::search::EmbeddedGeometryKind::LevelSetField:
            return make_context_level_set_sphere(center, radius, epoch, id);
        case svmp::search::EmbeddedGeometryKind::TriangulatedSurface:
            return make_context_triangulated_octahedron_sphere(center, radius, epoch, id);
        case svmp::search::EmbeddedGeometryKind::BooleanComposite: {
            auto active = make_context_sphere(center, radius, epoch + 1u, id + "-active");
            auto enclosing = make_context_sphere(center, 2.0, epoch + 2u, id + "-enclosing");
            return make_context_boolean_composite(
                svmp::search::EmbeddedGeometryBooleanOperation::Intersection,
                {active, enclosing},
                epoch,
                id);
        }
        default:
            return make_context_sphere(center, radius, epoch, id);
    }
}

void expect_cut_context_family_quadrature(
    const char* name,
    svmp::MeshBase mesh,
    const svmp::search::EmbeddedGeometryDescriptor& embedded,
    Real expected_parent_measure,
    Real expected_negative_measure,
    Real expected_positive_measure,
    Real expected_interface_measure)
{
    SCOPED_TRACE(name);
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto map = svmp::search::classify_embedded_geometry(mesh, embedded, options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map);
    const auto validity = svmp::search::diagnose_cut_topology_validity(topology);
    EXPECT_TRUE(validity.ok) << join_cut_validity_messages(validity.messages);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 123);

    ASSERT_EQ(context.volumeRules().size(), 2u);
    ASSERT_EQ(context.metadata().size(), 2u);
    ASSERT_EQ(context.bindings().size(), 2u);
    ASSERT_EQ(context.interfaceRules().size(), 1u);

    Real negative_measure = 0.0;
    Real positive_measure = 0.0;
    for (const auto& rule : context.volumeRules()) {
        EXPECT_EQ(rule.policy.kind, CutQuadratureConstructionKind::TopologySubdivision);
        EXPECT_EQ(rule.provenance.cut_topology_revision, topology.topology_revision);
        EXPECT_NEAR(rule.parent_measure, expected_parent_measure, 1.0e-10);
        EXPECT_FALSE(rule.points.empty());
        Real weight_sum = 0.0;
        for (const auto& qp : rule.points) {
            weight_sum += qp.weight;
        }
        EXPECT_NEAR(weight_sum, rule.measure, 1.0e-10);
        if (rule.side == CutIntegrationSide::Negative) {
            negative_measure += rule.measure;
        } else if (rule.side == CutIntegrationSide::Positive) {
            positive_measure += rule.measure;
        }
    }
    EXPECT_NEAR(negative_measure, expected_negative_measure, 1.0e-9);
    EXPECT_NEAR(positive_measure, expected_positive_measure, 1.0e-9);
    EXPECT_NEAR(negative_measure + positive_measure, expected_parent_measure, 1.0e-9);

    const auto& interface_rule = context.interfaceRules().front();
    EXPECT_EQ(interface_rule.policy.kind, CutQuadratureConstructionKind::TopologySubdivision);
    EXPECT_EQ(interface_rule.provenance.cut_topology_revision, topology.topology_revision);
    EXPECT_NEAR(interface_rule.measure, expected_interface_measure, 1.0e-9);
    ASSERT_FALSE(interface_rule.points.empty());
    EXPECT_NEAR(interface_rule.points.front().weight, expected_interface_measure, 1.0e-9);

    for (const auto& binding : context.bindings()) {
        EXPECT_EQ(binding.visible_to_paths.size(), 6u);
        EXPECT_EQ(binding.quadrature_policy_key, 123u);
    }
}

void expect_curved_isoparametric_context_import(
    const char* name,
    svmp::MeshBase mesh,
    const svmp::search::EmbeddedGeometryDescriptor& embedded)
{
    SCOPED_TRACE(name);
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto map = svmp::search::classify_embedded_geometry(mesh, embedded, options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    svmp::search::CutTopologyOptions topology_options;
    topology_options.allow_linearized_high_order_geometry = true;
    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map, topology_options);
    ASSERT_TRUE(topology.supported);
    ASSERT_FALSE(topology.curved_patches.empty());

    const auto validity = svmp::search::diagnose_cut_topology_validity(
        topology,
        /*high_order_parent_geometry=*/true);
    EXPECT_TRUE(validity.ok);
    EXPECT_TRUE(validity.requires_curved_geometry_support);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 456);

    ASSERT_GE(context.volumeRules().size(), 2u);
    Real volume_sum = 0.0;
    Real parent_measure = -1.0;
    bool saw_curved_volume = false;
    for (const auto& rule : context.volumeRules()) {
        volume_sum += rule.measure;
        if (parent_measure < 0.0) {
            parent_measure = rule.parent_measure;
        } else {
            EXPECT_NEAR(rule.parent_measure,
                        parent_measure,
                        std::max(Real{1.0e-8}, parent_measure * Real{1.0e-8}));
        }
        saw_curved_volume = saw_curved_volume ||
                            (rule.curved_geometry &&
                             rule.policy.kind == CutQuadratureConstructionKind::CurvedTopologySubdivision &&
                             !rule.points.empty());
    }
    ASSERT_GT(parent_measure, 0.0);
    EXPECT_NEAR(volume_sum,
                parent_measure,
                std::max(Real{1.0e-6}, parent_measure * Real{1.0e-6}));
    EXPECT_TRUE(saw_curved_volume);

    bool saw_curved_interface = false;
    for (const auto& rule : context.interfaceRules()) {
        saw_curved_interface = saw_curved_interface ||
                               (rule.curved_geometry &&
                                rule.policy.kind == CutQuadratureConstructionKind::CurvedTopologySubdivision &&
                                rule.measure > 0.0 &&
                                !rule.points.empty());
    }
    EXPECT_TRUE(saw_curved_interface);
}

const std::array<CutIntegrationAssemblyPath, 6>& all_cut_execution_paths()
{
    static const std::array<CutIntegrationAssemblyPath, 6> paths{{
        CutIntegrationAssemblyPath::Standard,
        CutIntegrationAssemblyPath::MatrixFree,
        CutIntegrationAssemblyPath::Interpreter,
        CutIntegrationAssemblyPath::AD,
        CutIntegrationAssemblyPath::SymbolicTangent,
        CutIntegrationAssemblyPath::JIT}};
    return paths;
}

bool binding_visible_to_path(const CutIntegrationBinding& binding,
                             CutIntegrationAssemblyPath path)
{
    return std::find(binding.visible_to_paths.begin(),
                     binding.visible_to_paths.end(),
                     path) != binding.visible_to_paths.end();
}

bool sensitivity_visible_to_path(const CutGeometrySensitivityMetadata& metadata,
                                 CutIntegrationAssemblyPath path)
{
    return std::find(metadata.visible_to_paths.begin(),
                     metadata.visible_to_paths.end(),
                     path) != metadata.visible_to_paths.end();
}

struct CutExecutionPathViewSummary {
    std::size_t binding_count{0};
    std::size_t volume_rule_count{0};
    std::size_t interface_rule_count{0};
    std::uint64_t topology_revision{0};
    std::uint64_t quadrature_policy_key{0};
    Real negative_measure{0.0};
    Real positive_measure{0.0};
    Real parent_measure{0.0};
    Real volume_fraction_sum{0.0};
    Real weighted_point_sum{0.0};
    Real weighted_normal_sum{0.0};
    Real generic_volume_integral{0.0};
    Real interface_measure{0.0};
    Real interface_normal_sum{0.0};
    std::vector<CutIntegrationSide> sides{};
    std::vector<std::string> provenance_ids{};
    std::vector<std::string> topology_ids{};
    std::vector<CutQuadratureConstructionKind> policies{};
    std::vector<std::size_t> point_counts{};
};

CutExecutionPathViewSummary summarize_cut_context_for_path(
    const CutIntegrationContext& context,
    CutIntegrationAssemblyPath path)
{
    CutExecutionPathViewSummary summary;
    summary.interface_rule_count = context.interfaceRules().size();
    for (const auto& interface_rule : context.interfaceRules()) {
        summary.interface_measure += interface_rule.measure;
        for (const auto& qp : interface_rule.points) {
            summary.interface_normal_sum +=
                qp.weight * (qp.normal[0] + Real{2.0} * qp.normal[1] +
                             Real{3.0} * qp.normal[2]);
        }
    }

    EXPECT_EQ(context.bindings().size(), context.volumeRules().size());
    EXPECT_EQ(context.metadata().size(), context.volumeRules().size());
    for (std::size_t i = 0; i < context.bindings().size(); ++i) {
        const auto& binding = context.bindings()[i];
        if (!binding_visible_to_path(binding, path)) {
            continue;
        }
        const auto& rule = context.volumeRules()[i];
        const auto& metadata = context.metadata()[i];
        ++summary.binding_count;
        ++summary.volume_rule_count;
        summary.topology_revision ^= binding.cut_topology_revision;
        summary.quadrature_policy_key ^= binding.quadrature_policy_key;
        summary.parent_measure = std::max(summary.parent_measure, rule.parent_measure);
        summary.volume_fraction_sum += metadata.volume_fraction;
        if (rule.side == CutIntegrationSide::Negative) {
            summary.negative_measure += rule.measure;
        } else if (rule.side == CutIntegrationSide::Positive) {
            summary.positive_measure += rule.measure;
        }
        for (const auto& qp : rule.points) {
            summary.weighted_point_sum +=
                qp.weight * (qp.point[0] + Real{2.0} * qp.point[1] +
                             Real{3.0} * qp.point[2]);
            summary.weighted_normal_sum +=
                qp.weight * (qp.normal[0] + Real{2.0} * qp.normal[1] +
                             Real{3.0} * qp.normal[2]);
            summary.generic_volume_integral +=
                qp.weight * (Real{1.0} + qp.point[0] +
                             Real{2.0} * qp.point[1] +
                             Real{3.0} * qp.point[2]);
        }
        summary.sides.push_back(rule.side);
        summary.provenance_ids.push_back(metadata.provenance_id);
        summary.topology_ids.push_back(metadata.cut_topology_id);
        summary.policies.push_back(rule.policy.kind);
        summary.point_counts.push_back(rule.points.size());
    }
    return summary;
}

void expect_equivalent_path_summary(
    const CutExecutionPathViewSummary& expected,
    const CutExecutionPathViewSummary& actual)
{
    EXPECT_EQ(actual.binding_count, expected.binding_count);
    EXPECT_EQ(actual.volume_rule_count, expected.volume_rule_count);
    EXPECT_EQ(actual.interface_rule_count, expected.interface_rule_count);
    EXPECT_EQ(actual.topology_revision, expected.topology_revision);
    EXPECT_EQ(actual.quadrature_policy_key, expected.quadrature_policy_key);
    EXPECT_NEAR(actual.negative_measure, expected.negative_measure, 1.0e-12);
    EXPECT_NEAR(actual.positive_measure, expected.positive_measure, 1.0e-12);
    EXPECT_NEAR(actual.parent_measure, expected.parent_measure, 1.0e-12);
    EXPECT_NEAR(actual.volume_fraction_sum, expected.volume_fraction_sum, 1.0e-12);
    EXPECT_NEAR(actual.weighted_point_sum, expected.weighted_point_sum, 1.0e-12);
    EXPECT_NEAR(actual.weighted_normal_sum, expected.weighted_normal_sum, 1.0e-12);
    EXPECT_NEAR(actual.generic_volume_integral, expected.generic_volume_integral, 1.0e-12);
    EXPECT_NEAR(actual.interface_measure, expected.interface_measure, 1.0e-12);
    EXPECT_NEAR(actual.interface_normal_sum, expected.interface_normal_sum, 1.0e-12);
    EXPECT_EQ(actual.sides, expected.sides);
    EXPECT_EQ(actual.provenance_ids, expected.provenance_ids);
    EXPECT_EQ(actual.topology_ids, expected.topology_ids);
    EXPECT_EQ(actual.policies, expected.policies);
    EXPECT_EQ(actual.point_counts, expected.point_counts);
}

[[nodiscard]] Real constant_cut_operator_integrand(const CutScalarOperatorPoint&)
{
    return Real{1.0};
}

[[nodiscard]] Real affine_cut_operator_integrand(const CutScalarOperatorPoint& qp)
{
    const Real side_indicator =
        qp.side == CutIntegrationSide::Negative
            ? Real{-1.0}
            : (qp.side == CutIntegrationSide::Positive ? Real{1.0} : Real{0.25});
    return Real{1.0} +
           qp.point[0] +
           Real{2.0} * qp.point[1] +
           Real{3.0} * qp.point[2] +
           Real{0.125} * side_indicator +
           Real{0.0625} * qp.volume_fraction +
           Real{1.0e-6} *
               static_cast<Real>(qp.cut_topology_revision % std::uint64_t{17});
}

[[nodiscard]] Real x_coordinate_cut_operator_integrand(const CutScalarOperatorPoint& qp)
{
    return qp.point[0];
}

void expect_equivalent_operator_evaluation(
    const CutScalarOperatorEvaluation& expected,
    const CutScalarOperatorEvaluation& actual,
    Real tolerance)
{
    EXPECT_EQ(actual.volume_rule_count, expected.volume_rule_count);
    EXPECT_EQ(actual.interface_rule_count, expected.interface_rule_count);
    EXPECT_EQ(actual.volume_point_count, expected.volume_point_count);
    EXPECT_EQ(actual.interface_point_count, expected.interface_point_count);
    EXPECT_NEAR(actual.parent_measure, expected.parent_measure, tolerance);
    EXPECT_NEAR(actual.negative_volume_measure,
                expected.negative_volume_measure,
                tolerance);
    EXPECT_NEAR(actual.positive_volume_measure,
                expected.positive_volume_measure,
                tolerance);
    EXPECT_NEAR(actual.interface_measure, expected.interface_measure, tolerance);
    EXPECT_NEAR(actual.negative_volume_integral,
                expected.negative_volume_integral,
                tolerance);
    EXPECT_NEAR(actual.positive_volume_integral,
                expected.positive_volume_integral,
                tolerance);
    EXPECT_NEAR(actual.interface_integral, expected.interface_integral, tolerance);
    EXPECT_NEAR(actual.volumeIntegral(), expected.volumeIntegral(), tolerance);
    EXPECT_NEAR(actual.totalIntegral(), expected.totalIntegral(), tolerance);
}

void expect_production_cut_operator_path_parity(
    const char* name,
    svmp::MeshBase mesh,
    const svmp::search::EmbeddedGeometryDescriptor& embedded,
    Real tolerance)
{
    SCOPED_TRACE(name);
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto map = svmp::search::classify_embedded_geometry(mesh, embedded, options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map);
    ASSERT_TRUE(topology.supported) << join_cut_validity_messages(topology.diagnostics);
    const auto validity = svmp::search::diagnose_cut_topology_validity(topology);
    ASSERT_TRUE(validity.ok) << join_cut_validity_messages(validity.messages);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 9100);

    const auto expected_view =
        summarize_cut_context_for_path(context, CutIntegrationAssemblyPath::Standard);
    const auto expected_constant = context.evaluateScalarCutOperator(
        CutIntegrationAssemblyPath::Standard,
        constant_cut_operator_integrand,
        constant_cut_operator_integrand);
    const auto expected_affine = context.evaluateScalarCutOperator(
        CutIntegrationAssemblyPath::Standard,
        affine_cut_operator_integrand,
        affine_cut_operator_integrand);

    ASSERT_GT(expected_constant.volume_rule_count, 0u);
    ASSERT_GT(expected_constant.interface_rule_count, 0u);
    ASSERT_GT(expected_constant.volume_point_count, 0u);
    ASSERT_GT(expected_constant.interface_point_count, 0u);
    EXPECT_EQ(expected_constant.volume_rule_count, expected_view.volume_rule_count);
    EXPECT_EQ(expected_constant.interface_rule_count, expected_view.interface_rule_count);
    EXPECT_NEAR(expected_constant.negative_volume_integral,
                expected_view.negative_measure,
                tolerance);
    EXPECT_NEAR(expected_constant.positive_volume_integral,
                expected_view.positive_measure,
                tolerance);
    EXPECT_NEAR(expected_constant.interface_integral,
                expected_view.interface_measure,
                tolerance);
    EXPECT_NEAR(expected_constant.volumeIntegral(),
                expected_constant.parent_measure,
                tolerance);
    EXPECT_NEAR(expected_constant.volumeIntegral(),
                expected_constant.negative_volume_measure +
                    expected_constant.positive_volume_measure,
                tolerance);
    EXPECT_NEAR(expected_constant.interface_integral,
                expected_constant.interface_measure,
                tolerance);

    for (const auto path : all_cut_execution_paths()) {
        const auto actual_constant = context.evaluateScalarCutOperator(
            path, constant_cut_operator_integrand, constant_cut_operator_integrand);
        expect_equivalent_operator_evaluation(
            expected_constant, actual_constant, tolerance);

        const auto actual_affine = context.evaluateScalarCutOperator(
            path, affine_cut_operator_integrand, affine_cut_operator_integrand);
        expect_equivalent_operator_evaluation(expected_affine, actual_affine, tolerance);
    }
}

void expect_curved_context_path_parity(
    const CutIntegrationContext& context,
    Real conservation_tolerance)
{
    const auto baseline =
        summarize_cut_context_for_path(context, CutIntegrationAssemblyPath::Standard);
    ASSERT_GT(baseline.binding_count, 0u);
    ASSERT_GT(baseline.volume_rule_count, 0u);
    ASSERT_GT(baseline.interface_rule_count, 0u);
    EXPECT_NEAR(baseline.negative_measure + baseline.positive_measure,
                baseline.parent_measure,
                std::max(conservation_tolerance,
                         baseline.parent_measure * conservation_tolerance));
    EXPECT_GT(baseline.interface_measure, 0.0);

    for (const auto path : all_cut_execution_paths()) {
        const auto actual = summarize_cut_context_for_path(context, path);
        expect_equivalent_path_summary(baseline, actual);
    }
}

void expect_linearized_context_path_parity(
    const char* name,
    svmp::MeshBase mesh,
    const svmp::search::EmbeddedGeometryDescriptor& embedded)
{
    SCOPED_TRACE(name);
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto map = svmp::search::classify_embedded_geometry(mesh, embedded, options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map);
    const auto validity = svmp::search::diagnose_cut_topology_validity(topology);
    ASSERT_TRUE(validity.ok) << join_cut_validity_messages(validity.messages);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 987);
    ASSERT_EQ(context.volumeRules().size(), 2u);
    ASSERT_EQ(context.bindings().size(), context.volumeRules().size());
    ASSERT_EQ(context.interfaceRules().size(), 1u);

    const auto baseline =
        summarize_cut_context_for_path(context, CutIntegrationAssemblyPath::Standard);
    ASSERT_EQ(baseline.binding_count, 2u);
    ASSERT_EQ(baseline.volume_rule_count, 2u);
    ASSERT_EQ(baseline.interface_rule_count, 1u);
    EXPECT_NEAR(baseline.negative_measure + baseline.positive_measure,
                baseline.parent_measure,
                1.0e-12);
    EXPECT_GT(baseline.interface_measure, 0.0);

    for (const auto path : all_cut_execution_paths()) {
        const auto actual = summarize_cut_context_for_path(context, path);
        expect_equivalent_path_summary(baseline, actual);
    }
}

void expect_linearized_plane_context_path_parity(
    const char* name,
    svmp::MeshBase mesh,
    const svmp::search::EmbeddedGeometryDescriptor& embedded)
{
    expect_linearized_context_path_parity(name, std::move(mesh), embedded);
}

void expect_quadratic_curved_plane_context_path_parity(
    const char* name,
    svmp::MeshBase mesh,
    const svmp::search::EmbeddedGeometryDescriptor& embedded)
{
    SCOPED_TRACE(name);
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto map = svmp::search::classify_embedded_geometry(mesh, embedded, options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    svmp::search::CutTopologyOptions topology_options;
    topology_options.allow_linearized_high_order_geometry = true;
    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map, topology_options);
    ASSERT_TRUE(topology.supported) << join_cut_validity_messages(topology.diagnostics);
    ASSERT_FALSE(topology.curved_patches.empty());
    EXPECT_TRUE(topology.linearized_cut_mode);

    const auto validity = svmp::search::diagnose_cut_topology_validity(
        topology,
        /*high_order_parent_geometry=*/true);
    ASSERT_TRUE(validity.ok) << join_cut_validity_messages(validity.messages);
    EXPECT_TRUE(validity.requires_curved_geometry_support);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 988);
    ASSERT_GE(context.volumeRules().size(), 2u);
    ASSERT_EQ(context.bindings().size(), context.volumeRules().size());
    ASSERT_FALSE(context.interfaceRules().empty());

    bool saw_curved_volume = false;
    for (const auto& rule : context.volumeRules()) {
        saw_curved_volume =
            saw_curved_volume ||
            (rule.curved_geometry &&
             rule.policy.kind == CutQuadratureConstructionKind::CurvedTopologySubdivision &&
             !rule.points.empty());
    }
    EXPECT_TRUE(saw_curved_volume);

    bool saw_curved_interface = false;
    for (const auto& rule : context.interfaceRules()) {
        saw_curved_interface =
            saw_curved_interface ||
            (rule.curved_geometry &&
             rule.policy.kind == CutQuadratureConstructionKind::CurvedTopologySubdivision &&
             rule.measure > 0.0 &&
             !rule.points.empty());
    }
    EXPECT_TRUE(saw_curved_interface);

    const auto baseline =
        summarize_cut_context_for_path(context, CutIntegrationAssemblyPath::Standard);
    ASSERT_GT(baseline.binding_count, 0u);
    EXPECT_NEAR(baseline.negative_measure + baseline.positive_measure,
                baseline.parent_measure,
                std::max(Real{1.0e-6}, baseline.parent_measure * Real{1.0e-6}));

    for (const auto path : all_cut_execution_paths()) {
        const auto actual = summarize_cut_context_for_path(context, path);
        expect_equivalent_path_summary(baseline, actual);
    }
}

void expect_sensitivity_metadata_shape_identities(
    const CutGeometrySensitivityMetadata& metadata)
{
    ASSERT_FALSE(metadata.parent_geometry_dofs.empty());
    ASSERT_FALSE(metadata.samples.empty());
    for (const auto& sample : metadata.samples) {
        ASSERT_EQ(sample.shape_values.size(), metadata.parent_geometry_dofs.size());
        ASSERT_EQ(sample.shape_gradients.size(), metadata.parent_geometry_dofs.size());
        Real value_sum = 0.0;
        std::array<Real, 3> gradient_sum{{0.0, 0.0, 0.0}};
        for (std::size_t i = 0; i < sample.shape_values.size(); ++i) {
            value_sum += sample.shape_values[i];
            gradient_sum[0] += sample.shape_gradients[i][0];
            gradient_sum[1] += sample.shape_gradients[i][1];
            gradient_sum[2] += sample.shape_gradients[i][2];
        }
        EXPECT_NEAR(value_sum, 1.0, 1.0e-11);
        EXPECT_NEAR(gradient_sum[0], 0.0, 1.0e-10);
        EXPECT_NEAR(gradient_sum[1], 0.0, 1.0e-10);
        EXPECT_NEAR(gradient_sum[2], 0.0, 1.0e-10);
    }
}

void expect_context_sensitivity_metadata(
    const CutIntegrationContext& context,
    const svmp::search::CutTopologyRecord& topology,
    std::uint64_t quadrature_policy_key,
    const std::string& construction_policy)
{
    ASSERT_FALSE(context.sensitivityMetadata().empty())
        << join_cut_validity_messages(topology.diagnostics);

    std::set<std::string> target_kinds;
    std::size_t matching_records = 0u;
    for (const auto& metadata : context.sensitivityMetadata()) {
        if (metadata.construction_policy != construction_policy) {
            continue;
        }
        ++matching_records;
        target_kinds.insert(metadata.target_kind);
        EXPECT_EQ(metadata.cut_topology_revision, topology.topology_revision);
        EXPECT_EQ(metadata.quadrature_policy_key, quadrature_policy_key);
        EXPECT_TRUE(metadata.ad_compatible);
        EXPECT_TRUE(metadata.location_sensitivity_available);
        EXPECT_TRUE(metadata.jacobian_sensitivity_available);
        EXPECT_TRUE(metadata.measure_sensitivity_available);
        EXPECT_TRUE(metadata.normal_sensitivity_available);
        EXPECT_TRUE(metadata.quadrature_weight_sensitivity_available);
        EXPECT_EQ(metadata.visible_to_paths.size(), all_cut_execution_paths().size());
        for (const auto path : all_cut_execution_paths()) {
            EXPECT_TRUE(sensitivity_visible_to_path(metadata, path));
        }
        expect_sensitivity_metadata_shape_identities(metadata);
    }
    EXPECT_GE(matching_records, 3u);
    EXPECT_NE(target_kinds.find("interface-quadrature"), target_kinds.end());
    EXPECT_NE(target_kinds.find("negative-volume-subcell-quadrature"), target_kinds.end());
    EXPECT_NE(target_kinds.find("positive-volume-subcell-quadrature"), target_kinds.end());
}

void expect_quadratic_curved_plane_sensitivity_path_visibility(
    const char* name,
    svmp::MeshBase mesh,
    const svmp::search::EmbeddedGeometryDescriptor& embedded)
{
    SCOPED_TRACE(name);
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto map = svmp::search::classify_embedded_geometry(mesh, embedded, options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    svmp::search::CutTopologyOptions topology_options;
    topology_options.allow_linearized_high_order_geometry = true;
    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map, topology_options);
    ASSERT_TRUE(topology.supported) << join_cut_validity_messages(topology.diagnostics);
    ASSERT_FALSE(topology.sensitivity_records.empty());

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 989);
    ASSERT_FALSE(context.sensitivityMetadata().empty());

    std::set<std::string> target_kinds;
    for (const auto& metadata : context.sensitivityMetadata()) {
        target_kinds.insert(metadata.target_kind);
        EXPECT_EQ(metadata.cut_topology_revision, topology.topology_revision);
        EXPECT_EQ(metadata.quadrature_policy_key, 989u);
        EXPECT_TRUE(metadata.ad_compatible);
        EXPECT_TRUE(metadata.location_sensitivity_available);
        EXPECT_TRUE(metadata.jacobian_sensitivity_available);
        EXPECT_TRUE(metadata.measure_sensitivity_available);
        EXPECT_TRUE(metadata.normal_sensitivity_available);
        EXPECT_TRUE(metadata.quadrature_weight_sensitivity_available);
        EXPECT_EQ(metadata.visible_to_paths.size(), all_cut_execution_paths().size());
        for (const auto path : all_cut_execution_paths()) {
            EXPECT_TRUE(sensitivity_visible_to_path(metadata, path));
        }
        expect_sensitivity_metadata_shape_identities(metadata);
    }
    EXPECT_NE(target_kinds.find("interface-quadrature"), target_kinds.end());
    EXPECT_NE(target_kinds.find("negative-volume-subcell-quadrature"), target_kinds.end());
    EXPECT_NE(target_kinds.find("positive-volume-subcell-quadrature"), target_kinds.end());

    std::vector<std::size_t> path_counts;
    for (const auto path : all_cut_execution_paths()) {
        std::size_t count = 0u;
        for (const auto& metadata : context.sensitivityMetadata()) {
            if (sensitivity_visible_to_path(metadata, path)) {
                ++count;
            }
        }
        path_counts.push_back(count);
    }
    ASSERT_FALSE(path_counts.empty());
    for (const auto count : path_counts) {
        EXPECT_EQ(count, path_counts.front());
    }
}

void expect_linearized_sensitivity_path_visibility(
    const char* name,
    svmp::MeshBase mesh,
    const svmp::search::EmbeddedGeometryDescriptor& embedded,
    std::uint64_t quadrature_policy_key)
{
    SCOPED_TRACE(name);
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto map = svmp::search::classify_embedded_geometry(mesh, embedded, options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map);
    ASSERT_TRUE(topology.supported) << join_cut_validity_messages(topology.diagnostics);
    ASSERT_FALSE(topology.sensitivity_records.empty())
        << join_cut_validity_messages(topology.diagnostics);

    CutIntegrationContext context;
    context.importCutTopology(topology,
                              CutGeometryFrame::Reference,
                              quadrature_policy_key);
    expect_context_sensitivity_metadata(context,
                                        topology,
                                        quadrature_policy_key,
                                        "linear-topology-subdivision");

    std::vector<std::size_t> path_counts;
    for (const auto path : all_cut_execution_paths()) {
        std::size_t count = 0u;
        for (const auto& metadata : context.sensitivityMetadata()) {
            if (metadata.construction_policy == "linear-topology-subdivision" &&
                sensitivity_visible_to_path(metadata, path)) {
                ++count;
            }
        }
        path_counts.push_back(count);
    }
    ASSERT_FALSE(path_counts.empty());
    for (const auto count : path_counts) {
        EXPECT_EQ(count, path_counts.front());
    }
}

} // namespace
#endif

TEST(CutIntegrationInfrastructure, ImportsGeneratedLevelSetInterfaceDomainByMarker)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/4,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/3);
    request.interface_marker = 51;
    request.quadrature_policy_key = 19;

    LevelSetInterfaceDomain domain(request);
    const LevelSetCellCutInput input{
        .parent_cell = 7,
        .element_type = ElementType::Quad4,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{1.0, 1.0, 0.0}},
                             {{0.0, 1.0, 0.0}}},
        .level_set_values = {-0.5, 0.5, 0.5, -0.5}};
    appendLinearLevelSetCellCut2D(domain, input);

    CutIntegrationContext context;
    context.addGeneratedInterfaceDomain(domain);

    EXPECT_TRUE(context.hasGeneratedInterfaceMarker(51));
    ASSERT_EQ(context.generatedInterfaceMarkers().size(), 1u);
    EXPECT_EQ(context.generatedInterfaceMarkers().front(), 51);
    ASSERT_EQ(context.interfaceRules().size(), 1u);
    ASSERT_EQ(context.volumeRules().size(), 2u);

    const auto marker_rules = context.interfaceRulesForMarker(51);
    ASSERT_EQ(marker_rules.size(), 1u);
    ASSERT_NE(marker_rules.front(), nullptr);
    EXPECT_EQ(marker_rules.front()->provenance.parent_entity, 7);
    EXPECT_EQ(marker_rules.front()->provenance.marker, 51);
    EXPECT_EQ(marker_rules.front()->provenance.predicate_policy_key, 19u);
    EXPECT_TRUE(context.interfaceRulesForMarker(52).empty());

    const auto negative_volume_rules =
        context.generatedVolumeRulesForMarkerAndSide(51, CutIntegrationSide::Negative);
    const auto positive_volume_rules =
        context.generatedVolumeRulesForMarkerAndSide(51, CutIntegrationSide::Positive);
    ASSERT_EQ(negative_volume_rules.size(), 1u);
    ASSERT_EQ(positive_volume_rules.size(), 1u);
    EXPECT_EQ(negative_volume_rules.front()->provenance.marker, 51);
    EXPECT_NEAR(negative_volume_rules.front()->measure, 0.5, 1.0e-14);
    EXPECT_NEAR(positive_volume_rules.front()->measure, 0.5, 1.0e-14);

    const auto evaluation = context.evaluateScalarCutOperator(
        CutIntegrationAssemblyPath::Standard,
        [](const CutScalarOperatorPoint&) { return 0.0; },
        [](const CutScalarOperatorPoint&) { return 2.0; });
    EXPECT_EQ(evaluation.interface_rule_count, 1u);
    EXPECT_EQ(evaluation.interface_point_count, 1u);
    EXPECT_DOUBLE_EQ(evaluation.interface_measure, 1.0);
    EXPECT_DOUBLE_EQ(evaluation.interface_integral, 2.0);

    context.clear();
    EXPECT_FALSE(context.hasGeneratedInterfaceMarker(51));
    EXPECT_FALSE(context.hasGeneratedVolumeMarker(51));
    EXPECT_TRUE(context.generatedInterfaceMarkers().empty());
    EXPECT_TRUE(context.generatedVolumeMarkers().empty());
    EXPECT_TRUE(context.interfaceRules().empty());
    EXPECT_TRUE(context.volumeRules().empty());
}

TEST(CutIntegrationInfrastructure, GeneratedCutVolumesRejectStaleSourceRevision)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/4,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/3);
    request.interface_marker = 51;

    LevelSetInterfaceDomain domain(request);
    const LevelSetCellCutInput input{
        .parent_cell = 7,
        .element_type = ElementType::Quad4,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{1.0, 1.0, 0.0}},
                             {{0.0, 1.0, 0.0}}},
        .level_set_values = {-0.5, 0.5, 0.5, -0.5}};
    appendLinearLevelSetCellCut2D(domain, input);

    CutIntegrationContext context;
    context.addGeneratedInterfaceDomain(domain);
    ASSERT_TRUE(context.hasExpectedGeneratedSourceValueRevision(51));
    EXPECT_EQ(context.expectedGeneratedSourceValueRevision(51), 3u);
    ASSERT_EQ(context.metadata().size(), 2u);
    EXPECT_EQ(context.metadata().front().source_value_revision, 3u);
    EXPECT_NO_THROW({
        const auto indices = context.generatedVolumeRuleIndicesForMarkerAndSide(
            51, CutIntegrationSide::Negative);
        (void)indices;
    });

    context.setExpectedGeneratedSourceValueRevision(51, 4u);
    EXPECT_THROW({
        const auto indices = context.generatedVolumeRuleIndicesForMarkerAndSide(
            51, CutIntegrationSide::Negative);
        (void)indices;
    }, std::invalid_argument);

    class CountingCellKernel final : public AssemblyKernel {
    public:
        [[nodiscard]] RequiredData getRequiredData() const override {
            return RequiredData::None;
        }

        void computeCell(const AssemblyContext&, KernelOutput&) override {}
    };

    CountingCellKernel kernel;
    CutDomainAssemblyOptions options;
    options.include_interface_rules = false;
    options.volume_marker = 51;
    options.volume_side = CutIntegrationSide::Negative;

    EXPECT_THROW({
        const auto summary = assembleCutDomains(
            context,
            kernel,
            [](const CutRuleAssemblyRequest&, AssemblyContext&) {},
            options);
        (void)summary;
    }, std::invalid_argument);
}

TEST(CutIntegrationInfrastructure, IndexesGeneratedLevelSetVolumeRulesByMarkerAndSide)
{
    CutIntegrationContext context;

    auto negative_rule = makeAxisAlignedBoxCutVolumeQuadrature(
        {{0.0, 0.0, 0.0}},
        {{1.0, 1.0, 1.0}},
        0,
        0.25,
        CutIntegrationSide::Negative,
        "level-set-water");
    negative_rule.provenance.parent_entity = 7;
    negative_rule.provenance.cut_topology_revision = 31;
    negative_rule.provenance.predicate_policy_key = 19;

    CutCellAssemblyMetadata negative_metadata;
    negative_metadata.cell = 7;
    negative_metadata.parent_entity = 7;
    negative_metadata.volume_fraction = negative_rule.volume_fraction;
    negative_metadata.side = CutIntegrationSide::Negative;
    negative_metadata.provenance_id = "level-set-water";
    negative_metadata.cut_topology_id = "generated-7-negative";
    negative_metadata.revision_key = 31;
    negative_metadata.cut_topology_revision = 31;
    negative_metadata.quadrature_policy_key = 19;
    context.addGeneratedVolumeRule(51, negative_metadata, negative_rule);

    auto positive_rule = makeAxisAlignedBoxCutVolumeQuadrature(
        {{0.0, 0.0, 0.0}},
        {{1.0, 1.0, 1.0}},
        0,
        0.25,
        CutIntegrationSide::Positive,
        "level-set-air");
    positive_rule.provenance.parent_entity = 7;
    positive_rule.provenance.cut_topology_revision = 31;
    positive_rule.provenance.predicate_policy_key = 19;

    CutCellAssemblyMetadata positive_metadata;
    positive_metadata.cell = 7;
    positive_metadata.parent_entity = 7;
    positive_metadata.volume_fraction = positive_rule.volume_fraction;
    positive_metadata.side = CutIntegrationSide::Positive;
    positive_metadata.provenance_id = "level-set-air";
    positive_metadata.cut_topology_id = "generated-7-positive";
    positive_metadata.revision_key = 31;
    positive_metadata.cut_topology_revision = 31;
    positive_metadata.quadrature_policy_key = 19;
    context.addGeneratedVolumeRule(51, positive_metadata, positive_rule);

    auto other_rule = makeAxisAlignedBoxCutVolumeQuadrature(
        {{0.0, 0.0, 0.0}},
        {{1.0, 1.0, 1.0}},
        1,
        0.5,
        CutIntegrationSide::Negative,
        "other-interface");
    other_rule.provenance.parent_entity = 8;
    CutCellAssemblyMetadata other_metadata;
    other_metadata.cell = 8;
    other_metadata.parent_entity = 8;
    other_metadata.volume_fraction = other_rule.volume_fraction;
    other_metadata.side = CutIntegrationSide::Negative;
    other_metadata.provenance_id = "other-interface";
    other_metadata.cut_topology_id = "generated-8-negative";
    context.addGeneratedVolumeRule(52, other_metadata, other_rule);

    EXPECT_TRUE(context.hasGeneratedVolumeMarker(51));
    EXPECT_TRUE(context.hasGeneratedVolumeMarker(52));
    EXPECT_FALSE(context.hasGeneratedVolumeMarker(53));
    ASSERT_EQ(context.generatedVolumeMarkers().size(), 2u);
    EXPECT_EQ(context.generatedVolumeMarkers()[0], 51);
    EXPECT_EQ(context.generatedVolumeMarkers()[1], 52);
    ASSERT_EQ(context.volumeRules().size(), 3u);
    ASSERT_EQ(context.metadata().size(), 3u);
    ASSERT_EQ(context.bindings().size(), context.volumeRules().size());
    EXPECT_TRUE(context.interfaceRules().empty());
    EXPECT_TRUE(context.generatedInterfaceMarkers().empty());

    const auto marker_rules = context.generatedVolumeRulesForMarker(51);
    ASSERT_EQ(marker_rules.size(), 2u);
    EXPECT_EQ(marker_rules[0]->side, CutIntegrationSide::Negative);
    EXPECT_EQ(marker_rules[1]->side, CutIntegrationSide::Positive);

    const auto negative_rules =
        context.generatedVolumeRulesForMarkerAndSide(51, CutIntegrationSide::Negative);
    ASSERT_EQ(negative_rules.size(), 1u);
    ASSERT_NE(negative_rules.front(), nullptr);
    EXPECT_EQ(negative_rules.front()->provenance.parent_entity, 7);
    EXPECT_EQ(negative_rules.front()->provenance.predicate_policy_key, 19u);
    EXPECT_NEAR(negative_rules.front()->measure, 0.25, 1.0e-14);

    const auto positive_rules =
        context.generatedVolumeRulesForMarkerAndSide(51, CutIntegrationSide::Positive);
    ASSERT_EQ(positive_rules.size(), 1u);
    ASSERT_NE(positive_rules.front(), nullptr);
    EXPECT_EQ(positive_rules.front()->side, CutIntegrationSide::Positive);
    EXPECT_NEAR(positive_rules.front()->measure, 0.75, 1.0e-14);

    EXPECT_TRUE(context.generatedVolumeRulesForMarkerAndSide(
                           51, CutIntegrationSide::Interface)
                    .empty());
    EXPECT_TRUE(context.generatedVolumeRulesForMarkerAndSide(
                           53, CutIntegrationSide::Negative)
                    .empty());

    const auto negative_indices =
        context.generatedVolumeRuleIndicesForMarkerAndSide(51, CutIntegrationSide::Negative);
    ASSERT_EQ(negative_indices.size(), 1u);
    EXPECT_EQ(context.metadata()[negative_indices.front()].parent_entity, 7);
    EXPECT_EQ(context.metadata()[negative_indices.front()].side, CutIntegrationSide::Negative);

    const auto negative_metadata_view =
        context.generatedVolumeMetadataForMarkerAndSide(51, CutIntegrationSide::Negative);
    ASSERT_EQ(negative_metadata_view.size(), 1u);
    ASSERT_NE(negative_metadata_view.front(), nullptr);
    EXPECT_DOUBLE_EQ(negative_metadata_view.front()->volume_fraction, 0.25);
    EXPECT_EQ(negative_metadata_view.front()->cut_topology_id, "generated-7-negative");

    CutCellAssemblyMetadata mismatched_metadata = negative_metadata;
    mismatched_metadata.side = CutIntegrationSide::Positive;
    EXPECT_THROW(context.addGeneratedVolumeRule(51, mismatched_metadata, negative_rule),
                 std::invalid_argument);

    context.clear();
    EXPECT_FALSE(context.hasGeneratedVolumeMarker(51));
    EXPECT_TRUE(context.generatedVolumeMarkers().empty());
    EXPECT_TRUE(context.generatedVolumeRulesForMarker(51).empty());
    EXPECT_TRUE(context.volumeRules().empty());
    EXPECT_TRUE(context.metadata().empty());
}

TEST(CutIntegrationInfrastructure, CutDomainAssemblerFiltersGeneratedVolumeMarkerAndSide)
{
    class CountingCellKernel final : public AssemblyKernel {
    public:
        [[nodiscard]] RequiredData getRequiredData() const override {
            return RequiredData::None;
        }

        void computeCell(const AssemblyContext&, KernelOutput& output) override {
            output.reserve(/*n_test=*/1, /*n_trial=*/1,
                           /*need_matrix=*/false,
                           /*need_vector=*/true);
            output.vectorEntry(0) = Real{1.0};
        }
    };

    auto make_domain = [](int marker, MeshIndex parent_cell) {
        CutInterfaceDomainRequest request;
        request.source = LevelSetInterfaceSource::fromField(
            /*field_id=*/static_cast<FieldId>(marker),
            /*layout_revision=*/1,
            /*value_revision=*/1);
        request.interface_marker = marker;

        LevelSetInterfaceDomain domain(request);
        const LevelSetCellCutInput input{
            .parent_cell = parent_cell,
            .element_type = ElementType::Quad4,
            .node_coordinates = {{{0.0, 0.0, 0.0}},
                                 {{1.0, 0.0, 0.0}},
                                 {{1.0, 1.0, 0.0}},
                                 {{0.0, 1.0, 0.0}}},
            .level_set_values = {-0.5, 0.5, 0.5, -0.5}};
        appendLinearLevelSetCellCut2D(domain, input);
        return domain;
    };

    CutIntegrationContext context;
    context.addGeneratedInterfaceDomain(make_domain(51, 7));
    context.addGeneratedInterfaceDomain(make_domain(52, 8));

    CountingCellKernel kernel;
    std::vector<std::size_t> seen_indices;
    std::vector<int> seen_markers;
    std::vector<CutIntegrationSide> seen_sides;

    CutDomainAssemblyOptions options;
    options.include_interface_rules = false;
    options.volume_marker = 51;
    options.volume_side = CutIntegrationSide::Negative;

    const auto summary = assembleCutDomains(
        context,
        kernel,
        [&](const CutRuleAssemblyRequest& request, AssemblyContext&) {
            ASSERT_NE(request.rule, nullptr);
            seen_indices.push_back(request.rule_index);
            seen_markers.push_back(request.marker);
            seen_sides.push_back(request.rule->side);
        },
        options);

    EXPECT_EQ(summary.volume_rule_count, 1u);
    EXPECT_EQ(summary.interface_rule_count, 0u);
    EXPECT_EQ(summary.skipped_rule_count, 3u);
    ASSERT_TRUE(summary.hasVector());
    ASSERT_EQ(summary.total_output.local_vector.size(), 1u);
    EXPECT_DOUBLE_EQ(summary.total_output.local_vector[0], 1.0);

    ASSERT_EQ(seen_indices.size(), 1u);
    ASSERT_EQ(seen_markers.size(), 1u);
    ASSERT_EQ(seen_sides.size(), 1u);
    EXPECT_EQ(seen_markers.front(), 51);
    EXPECT_EQ(seen_sides.front(), CutIntegrationSide::Negative);
    EXPECT_EQ(context.volumeRules()[seen_indices.front()].provenance.marker, 51);
    EXPECT_EQ(context.volumeRules()[seen_indices.front()].side, CutIntegrationSide::Negative);
}

TEST(CutIntegrationInfrastructure, GeneratedLevelSetInterfaceIntegratesConstantsAcrossPaths)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/5,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/1);
    request.interface_marker = 52;

    LevelSetInterfaceDomain domain(request);
    const LevelSetCellCutInput input{
        .parent_cell = 8,
        .element_type = ElementType::Quad4,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{1.0, 1.0, 0.0}},
                             {{0.0, 1.0, 0.0}}},
        .level_set_values = {-0.5, 0.5, 0.5, -0.5}};
    appendLinearLevelSetCellCut2D(domain, input);

    CutIntegrationContext context;
    context.addGeneratedInterfaceDomain(domain);

    const std::array<CutIntegrationAssemblyPath, 6> paths{{
        CutIntegrationAssemblyPath::Standard,
        CutIntegrationAssemblyPath::MatrixFree,
        CutIntegrationAssemblyPath::Interpreter,
        CutIntegrationAssemblyPath::AD,
        CutIntegrationAssemblyPath::SymbolicTangent,
        CutIntegrationAssemblyPath::JIT}};
    for (const auto path : paths) {
        const auto evaluation = context.evaluateScalarCutOperator(
            path,
            [](const CutScalarOperatorPoint&) { return Real{1.0}; },
            [](const CutScalarOperatorPoint&) { return Real{1.0}; });

        EXPECT_EQ(evaluation.volume_rule_count, 2u);
        EXPECT_GE(evaluation.volume_point_count, evaluation.volume_rule_count);
        EXPECT_NEAR(evaluation.negative_volume_measure, 0.5, 1.0e-14);
        EXPECT_NEAR(evaluation.positive_volume_measure, 0.5, 1.0e-14);
        EXPECT_NEAR(evaluation.volumeIntegral(), 1.0, 1.0e-14);
        EXPECT_EQ(evaluation.interface_rule_count, 1u);
        EXPECT_EQ(evaluation.interface_point_count, 1u);
        EXPECT_NEAR(evaluation.interface_measure, 1.0, 1.0e-14);
        EXPECT_NEAR(evaluation.interface_integral, 1.0, 1.0e-14);
        EXPECT_NEAR(evaluation.totalIntegral(), 2.0, 1.0e-14);
    }
}

TEST(CutIntegrationInfrastructure, GeneratedLevelSetInterfaceIntegratesLinearFieldsAcrossPaths)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/6,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/1);
    request.interface_marker = 53;

    LevelSetInterfaceDomain domain(request);
    const LevelSetCellCutInput input{
        .parent_cell = 9,
        .element_type = ElementType::Quad4,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{1.0, 1.0, 0.0}},
                             {{0.0, 1.0, 0.0}}},
        .level_set_values = {-0.5, 0.5, 0.5, -0.5}};
    appendLinearLevelSetCellCut2D(domain, input);

    CutIntegrationContext context;
    context.addGeneratedInterfaceDomain(domain);

    const std::array<CutIntegrationAssemblyPath, 6> paths{{
        CutIntegrationAssemblyPath::Standard,
        CutIntegrationAssemblyPath::MatrixFree,
        CutIntegrationAssemblyPath::Interpreter,
        CutIntegrationAssemblyPath::AD,
        CutIntegrationAssemblyPath::SymbolicTangent,
        CutIntegrationAssemblyPath::JIT}};
    for (const auto path : paths) {
        const auto evaluation = context.evaluateScalarCutOperator(
            path,
            [](const CutScalarOperatorPoint& point) {
                return point.point[0] + Real{2.0} * point.point[1];
            },
            [](const CutScalarOperatorPoint& point) {
                return point.point[0] + Real{2.0} * point.point[1];
            });

        EXPECT_EQ(evaluation.volume_rule_count, 2u);
        EXPECT_EQ(evaluation.interface_rule_count, 1u);
        EXPECT_EQ(evaluation.interface_point_count, 1u);
        EXPECT_NEAR(evaluation.volumeIntegral(), 1.5, 1.0e-14);
        EXPECT_NEAR(evaluation.interface_measure, 1.0, 1.0e-14);
        EXPECT_NEAR(evaluation.interface_integral, 1.5, 1.0e-14);
        EXPECT_NEAR(evaluation.totalIntegral(), 3.0, 1.0e-14);
    }
}

TEST(CutIntegrationInfrastructure, AssemblyContextCarriesQuadratureMetadataAndHooks)
{
    CutIntegrationContext context;
    CutCellAssemblyMetadata metadata;
    metadata.cell = 4;
    metadata.volume_fraction = 0.5;
    metadata.embedded_normal = {{1.0, 0.0, 0.0}};
    metadata.provenance_id = "embedded-plane";
    metadata.revision_key = 99;

    auto rule = makeAxisAlignedBoxCutVolumeQuadrature(
        {{0.0, 0.0, 0.0}},
        {{1.0, 1.0, 1.0}},
        0,
        0.5,
        CutIntegrationSide::Negative,
        "embedded-plane");
    context.addVolumeRule(metadata, rule);
    context.addInterfaceRule(makeAxisAlignedBoxCutInterfaceQuadrature(
        {{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}, 0, 0.5, "embedded-plane"));

    EmbeddedBoundaryKinematicData kinematics;
    kinematics.constraint_id = "moving-plane";
    kinematics.relation_map_id = "relation";
    kinematics.source_geometry_id = "plane";
    kinematics.provenance_id = "embedded-plane";
    context.addKinematicData(kinematics);

    CutStabilizationHook hook;
    hook.name = "aggregation-candidate";
    hook.geometry_scale = 0.5;
    hook.conditioning_indicator = 1.0e-9;
    hook.enabled = true;
    context.addStabilizationHook(hook);

    ASSERT_EQ(context.metadata().size(), 1u);
    ASSERT_EQ(context.volumeRules().size(), 1u);
    ASSERT_EQ(context.interfaceRules().size(), 1u);
    ASSERT_EQ(context.kinematicData().size(), 1u);
    ASSERT_EQ(context.stabilizationHooks().size(), 1u);
    context.addBinding({4,
                        CutQuadratureKind::Volume,
                        CutIntegrationSide::Negative,
                        99,
                        100,
                        101,
                        102,
                        {CutIntegrationAssemblyPath::Standard,
                         CutIntegrationAssemblyPath::MatrixFree,
                         CutIntegrationAssemblyPath::Interpreter,
                         CutIntegrationAssemblyPath::AD,
                         CutIntegrationAssemblyPath::SymbolicTangent,
                         CutIntegrationAssemblyPath::JIT}});
    ASSERT_EQ(context.bindings().size(), 1u);
    EXPECT_EQ(context.bindings()[0].visible_to_paths.size(), 6u);
    EXPECT_DOUBLE_EQ(context.volumeRules()[0].measure, 0.5);
    EXPECT_EQ(context.metadata()[0].provenance_id, "embedded-plane");
}

TEST(CutIntegrationInfrastructure, InvalidationSeparatesGeometryAndFELayoutChanges)
{
    CutIntegrationRevisionSnapshot cached;
    cached.valid = true;
    cached.cut_revision_key = 10;
    cached.geometry_revision = 1;
    cached.fe_dof_layout_revision = 2;
    cached.cut_cell_count = 1;

    auto current = cached;
    EXPECT_FALSE(classifyCutIntegrationRefresh(cached, current).any());

    current.geometry_revision = 2;
    auto geometry_decision = classifyCutIntegrationRefresh(cached, current);
    EXPECT_TRUE(geometry_decision.rebuild_cut_classification);
    EXPECT_TRUE(geometry_decision.rebuild_quadrature);
    EXPECT_FALSE(geometry_decision.rebuild_sparsity_pattern);
    EXPECT_TRUE(geometry_decision.refresh_full_cell_domain_caches);
    EXPECT_TRUE(geometry_decision.rebuild_matrix_free_data);

    current = cached;
    current.cut_topology_revision = 77;
    auto topology_decision = classifyCutIntegrationRefresh(cached, current);
    EXPECT_TRUE(topology_decision.rebuild_quadrature);
    EXPECT_TRUE(topology_decision.rebuild_sparsity_pattern);
    EXPECT_FALSE(topology_decision.refresh_full_cell_domain_caches);
    EXPECT_TRUE(topology_decision.update_stabilization_hooks);

    current = cached;
    current.embedded_field_value_revision = 9;
    auto field_decision = classifyCutIntegrationRefresh(cached, current);
    EXPECT_TRUE(field_decision.rebuild_cut_classification);
    EXPECT_TRUE(field_decision.rebuild_sparsity_pattern);
    EXPECT_FALSE(field_decision.refresh_full_cell_domain_caches);

    current = cached;
    current.conditioning_revision = 4;
    auto conditioning_decision = classifyCutIntegrationRefresh(cached, current);
    EXPECT_TRUE(conditioning_decision.rebuild_sparsity_pattern);
    EXPECT_FALSE(conditioning_decision.refresh_full_cell_domain_caches);
    EXPECT_TRUE(conditioning_decision.update_stabilization_hooks);
    EXPECT_TRUE(conditioning_decision.refresh_preconditioner);

    current = cached;
    current.fe_dof_layout_revision = 3;
    auto layout_decision = classifyCutIntegrationRefresh(cached, current);
    EXPECT_FALSE(layout_decision.rebuild_cut_classification);
    EXPECT_FALSE(layout_decision.rebuild_quadrature);
    EXPECT_TRUE(layout_decision.rebuild_sparsity_pattern);
    EXPECT_TRUE(layout_decision.refresh_full_cell_domain_caches);
    EXPECT_TRUE(layout_decision.rebuild_matrix);
    EXPECT_TRUE(layout_decision.refresh_preconditioner);
}

TEST(CutIntegrationInfrastructure, InvalidationCoversCutCacheDependencyClasses)
{
    CutIntegrationRevisionSnapshot cached;
    cached.valid = true;
    cached.cut_revision_key = 10;
    cached.geometry_revision = 1;
    cached.active_configuration_epoch = 2;
    cached.cut_topology_revision = 3;
    cached.quadrature_policy_revision = 4;
    cached.conditioning_revision = 5;
    cached.fe_space_revision = 6;
    cached.fe_dof_layout_revision = 7;
    cached.cut_cell_count = 2;
    cached.cut_face_count = 1;

    auto marker_side_change = cached;
    marker_side_change.active_configuration_epoch = 20;
    const auto marker_side_decision =
        classifyCutIntegrationRefresh(cached, marker_side_change);
    EXPECT_TRUE(marker_side_decision.rebuild_cut_classification);
    EXPECT_TRUE(marker_side_decision.rebuild_quadrature);
    EXPECT_TRUE(marker_side_decision.rebuild_sparsity_pattern);
    EXPECT_FALSE(marker_side_decision.refresh_full_cell_domain_caches);
    EXPECT_TRUE(marker_side_decision.rebuild_matrix);
    EXPECT_TRUE(marker_side_decision.rebuild_matrix_free_data);
    EXPECT_TRUE(marker_side_decision.refresh_preconditioner);
    EXPECT_TRUE(marker_side_decision.refresh_restart_metadata);
    EXPECT_TRUE(marker_side_decision.update_stabilization_hooks);

    auto cut_volume_rule_change = cached;
    cut_volume_rule_change.quadrature_policy_revision = 40;
    const auto cut_rule_decision =
        classifyCutIntegrationRefresh(cached, cut_volume_rule_change);
    EXPECT_TRUE(cut_rule_decision.rebuild_quadrature);
    EXPECT_FALSE(cut_rule_decision.rebuild_sparsity_pattern);
    EXPECT_FALSE(cut_rule_decision.refresh_full_cell_domain_caches);
    EXPECT_TRUE(cut_rule_decision.rebuild_matrix);
    EXPECT_TRUE(cut_rule_decision.rebuild_matrix_free_data);
    EXPECT_TRUE(cut_rule_decision.refresh_preconditioner);

    auto interface_rule_change = cached;
    interface_rule_change.cut_topology_revision = 30;
    const auto interface_decision =
        classifyCutIntegrationRefresh(cached, interface_rule_change);
    EXPECT_TRUE(interface_decision.rebuild_quadrature);
    EXPECT_TRUE(interface_decision.rebuild_sparsity_pattern);
    EXPECT_FALSE(interface_decision.refresh_full_cell_domain_caches);
    EXPECT_TRUE(interface_decision.rebuild_matrix);
    EXPECT_TRUE(interface_decision.rebuild_matrix_free_data);

    auto cut_facet_change = cached;
    cut_facet_change.conditioning_revision = 50;
    const auto cut_facet_decision =
        classifyCutIntegrationRefresh(cached, cut_facet_change);
    EXPECT_TRUE(cut_facet_decision.rebuild_sparsity_pattern);
    EXPECT_FALSE(cut_facet_decision.refresh_full_cell_domain_caches);
    EXPECT_TRUE(cut_facet_decision.rebuild_matrix);
    EXPECT_TRUE(cut_facet_decision.rebuild_matrix_free_data);
    EXPECT_TRUE(cut_facet_decision.update_stabilization_hooks);

    auto fe_layout_change = cached;
    fe_layout_change.fe_space_revision = 60;
    const auto layout_decision =
        classifyCutIntegrationRefresh(cached, fe_layout_change);
    EXPECT_FALSE(layout_decision.rebuild_cut_classification);
    EXPECT_FALSE(layout_decision.rebuild_quadrature);
    EXPECT_TRUE(layout_decision.rebuild_sparsity_pattern);
    EXPECT_TRUE(layout_decision.refresh_full_cell_domain_caches);
    EXPECT_FALSE(layout_decision.update_stabilization_hooks);
    EXPECT_TRUE(layout_decision.rebuild_matrix);
}

TEST(CutIntegrationInfrastructure, ConditioningDiagnosticsIdentifySmallAndDegenerateCuts)
{
    const auto diagnostic = diagnoseCutConditioning({0.5, 1.0e-9, 0.0}, 1.0e-6, 1.0e-12);
    EXPECT_FALSE(diagnostic.ok);
    EXPECT_EQ(diagnostic.small_cut_cell_count, 1u);
    EXPECT_EQ(diagnostic.degenerate_cut_count, 1u);
}

TEST(CutIntegrationInfrastructure, SmallCutNeighborhoodsAreDeterministicPhysicsNeutralHooks)
{
    const auto neighborhoods = buildCutConditioningNeighborhoods(
        {2, 4, 6},
        {0.5, 1.0e-10, 1.0e-9},
        {{4, 2}, {4, 6}, {6, 8}},
        1.0e-8);

    ASSERT_EQ(neighborhoods.size(), 2u);
    EXPECT_EQ(neighborhoods[0].cut_cell, 4);
    EXPECT_EQ(neighborhoods[0].adjacent_cells.size(), 2u);
    EXPECT_EQ(neighborhoods[0].extension_patch.front(), 4);
    EXPECT_GT(neighborhoods[0].conditioning_indicator, 1.0e8);
    EXPECT_NE(neighborhoods[0].stable_id, 0u);
    EXPECT_EQ(neighborhoods[1].cut_cell, 6);
}

TEST(CutIntegrationInfrastructure, IdentifiesInteriorFacetsAdjacentToCutCells)
{
    const auto facets = identifyCutAdjacentInteriorFacets(
        {6, 2, 6},
        {CutInteriorFacetAdjacency{.facet = 20, .first_cell = 1, .second_cell = 2},
         CutInteriorFacetAdjacency{.facet = 21, .first_cell = 2, .second_cell = 6},
         CutInteriorFacetAdjacency{.facet = 22, .first_cell = 3, .second_cell = 4},
         CutInteriorFacetAdjacency{.facet = 23, .first_cell = 6, .second_cell = 8},
         CutInteriorFacetAdjacency{.facet = 24, .first_cell = 6, .second_cell = -1}});

    ASSERT_EQ(facets.size(), 3u);
    EXPECT_EQ(facets[0].facet, 20);
    EXPECT_FALSE(facets[0].first_cell_cut);
    EXPECT_TRUE(facets[0].second_cell_cut);
    EXPECT_EQ(facets[1].facet, 21);
    EXPECT_TRUE(facets[1].first_cell_cut);
    EXPECT_TRUE(facets[1].second_cell_cut);
    EXPECT_EQ(facets[2].facet, 23);
    EXPECT_TRUE(facets[2].first_cell_cut);
    EXPECT_FALSE(facets[2].second_cell_cut);
    EXPECT_NE(facets[2].stable_id, 0u);
}

TEST(CutIntegrationInfrastructure, BuildsMarkerBackedCutAdjacentFacetSetHandle)
{
    const auto facets = identifyCutAdjacentInteriorFacets(
        {2, 6},
        {CutInteriorFacetAdjacency{.facet = 20, .first_cell = 1, .second_cell = 2},
         CutInteriorFacetAdjacency{.facet = 21, .first_cell = 2, .second_cell = 6},
         CutInteriorFacetAdjacency{.facet = 20, .first_cell = 1, .second_cell = 2},
         CutInteriorFacetAdjacency{.facet = 23, .first_cell = 6, .second_cell = 8}});

    const auto handle =
        makeCutAdjacentFacetSetHandle(/*marker=*/130, "cut-adjacent-facets", facets);

    EXPECT_TRUE(handle.valid());
    EXPECT_EQ(handle.marker, 130);
    EXPECT_EQ(handle.name, "cut-adjacent-facets");
    ASSERT_EQ(handle.facets.size(), 3u);
    EXPECT_EQ(handle.facets[0], 20);
    EXPECT_EQ(handle.facets[1], 21);
    EXPECT_EQ(handle.facets[2], 23);
    ASSERT_EQ(handle.facet_metadata.size(), 3u);
    EXPECT_EQ(handle.facet_metadata[0].facet, 20);
    EXPECT_EQ(handle.facet_metadata[0].first_cell, 1);
    EXPECT_EQ(handle.facet_metadata[0].second_cell, 2);
    EXPECT_EQ(handle.facet_metadata[1].facet, 21);
    EXPECT_EQ(handle.facet_metadata[1].first_cell, 2);
    EXPECT_EQ(handle.facet_metadata[1].second_cell, 6);
    EXPECT_EQ(handle.facet_metadata[2].facet, 23);
    EXPECT_EQ(handle.facet_metadata[2].first_cell, 6);
    EXPECT_EQ(handle.facet_metadata[2].second_cell, 8);
    EXPECT_NE(handle.stable_id, 0u);
}

TEST(CutIntegrationInfrastructure, BuildsCutAdjacentFacetSetFromGeneratedInterfaceDomain)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/10,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/1);
    request.interface_marker = 54;

    LevelSetInterfaceDomain domain(request);
    const LevelSetCellCutInput cut_input{
        .element_type = ElementType::Quad4,
        .node_coordinates = {{{0.0, 0.0, 0.0}},
                             {{1.0, 0.0, 0.0}},
                             {{1.0, 1.0, 0.0}},
                             {{0.0, 1.0, 0.0}}},
        .level_set_values = {-0.5, 0.5, 0.5, -0.5}};
    auto cell_two_input = cut_input;
    cell_two_input.parent_cell = 2;
    appendLinearLevelSetCellCut2D(domain, cell_two_input);
    auto cell_six_input = cut_input;
    cell_six_input.parent_cell = 6;
    appendLinearLevelSetCellCut2D(domain, cell_six_input);

    const auto adjacent_facets = identifyCutAdjacentInteriorFacets(
        domain.cutCells(),
        {CutInteriorFacetAdjacency{.facet = 20, .first_cell = 1, .second_cell = 2},
         CutInteriorFacetAdjacency{.facet = 21, .first_cell = 2, .second_cell = 6},
         CutInteriorFacetAdjacency{.facet = 22, .first_cell = 3, .second_cell = 4},
         CutInteriorFacetAdjacency{.facet = 23, .first_cell = 6, .second_cell = 8}});
    const auto handle =
        makeCutAdjacentFacetSetHandle(/*marker=*/131, "generated-cut-facets", adjacent_facets);

    ASSERT_TRUE(handle.valid());
    ASSERT_EQ(handle.facets.size(), 3u);
    EXPECT_EQ(handle.facets[0], 20);
    EXPECT_EQ(handle.facets[1], 21);
    EXPECT_EQ(handle.facets[2], 23);

    CutIntegrationContext context;
    context.addGeneratedInterfaceDomain(domain);
    CutFacetSetHandle context_handle;
    context_handle.marker = handle.marker;
    context_handle.name = handle.name;
    context_handle.facets = {handle.facets[2], handle.facets[0], handle.facets[1], handle.facets[0]};
    for (const auto& facet : handle.facet_metadata) {
        CutFacetSetFacetMetadata metadata;
        metadata.facet = facet.facet;
        metadata.first_cell = facet.first_cell;
        metadata.second_cell = facet.second_cell;
        metadata.stabilization_scale = facet.stabilization_scale;
        metadata.stable_id = facet.stable_id;
        context_handle.facet_metadata.push_back(metadata);
    }
    context_handle.stable_id = handle.stable_id;
    context.addFacetSetHandle(std::move(context_handle));

    EXPECT_TRUE(context.hasFacetSetHandleMarker(131));
    ASSERT_EQ(context.facetSetHandles().size(), 1u);
    const auto* stored = context.facetSetHandleForMarker(131);
    ASSERT_NE(stored, nullptr);
    EXPECT_TRUE(stored->valid());
    EXPECT_EQ(stored->marker, 131);
    EXPECT_EQ(stored->name, "generated-cut-facets");
    ASSERT_EQ(stored->facets.size(), 3u);
    EXPECT_EQ(stored->facets[0], 20);
    EXPECT_EQ(stored->facets[1], 21);
    EXPECT_EQ(stored->facets[2], 23);
    EXPECT_TRUE(stored->containsFacet(20));
    EXPECT_TRUE(stored->containsFacet(21));
    EXPECT_TRUE(stored->containsFacet(23));
    EXPECT_FALSE(stored->containsFacet(22));
    ASSERT_EQ(stored->facet_metadata.size(), 3u);
    EXPECT_NEAR(stored->stabilizationScaleForFacet(20), 2.0, 1.0e-12);
    EXPECT_NEAR(stored->stabilizationScaleForFacet(21), 2.0, 1.0e-12);
    EXPECT_NEAR(stored->stabilizationScaleForFacet(23), 2.0, 1.0e-12);
    EXPECT_DOUBLE_EQ(stored->stabilizationScaleForFacet(22), 0.0);
    EXPECT_NE(stored->stable_id, 0u);

    context.clear();
    EXPECT_FALSE(context.hasFacetSetHandleMarker(131));
    EXPECT_EQ(context.facetSetHandleForMarker(131), nullptr);
}

TEST(CutIntegrationInfrastructure, CapsCutAdjacentFacetStabilizationScales)
{
    CutIntegrationContext context;

    CutCellAssemblyMetadata small_fragment;
    small_fragment.cell = 4;
    small_fragment.parent_entity = 4;
    small_fragment.volume_fraction = 1.0e-10;
    small_fragment.side = CutIntegrationSide::Negative;

    CutQuadratureRule small_fragment_rule;
    small_fragment_rule.side = CutIntegrationSide::Negative;
    small_fragment_rule.measure = small_fragment.volume_fraction;
    small_fragment_rule.parent_measure = 1.0;
    small_fragment_rule.volume_fraction = small_fragment.volume_fraction;
    context.addVolumeRule(small_fragment, small_fragment_rule);

    CutFacetSetHandle handle;
    handle.marker = 132;
    handle.name = "capped-cut-facets";
    handle.facets = {20, 21};
    handle.facet_metadata = {
        CutFacetSetFacetMetadata{.facet = 20,
                                  .first_cell = 4,
                                  .second_cell = 5,
                                  .stabilization_scale = 0.0,
                                  .stable_id = 1u},
        CutFacetSetFacetMetadata{.facet = 21,
                                  .first_cell = 6,
                                  .second_cell = 7,
                                  .stabilization_scale = 1.0e12,
                                  .stable_id = 2u}};

    const auto& stored = context.addFacetSetHandle(std::move(handle));
    EXPECT_DOUBLE_EQ(stored.stabilizationScaleForFacet(20),
                     CutIntegrationContext::maxCutCellStabilizationScale());
    EXPECT_DOUBLE_EQ(stored.stabilizationScaleForFacet(21),
                     CutIntegrationContext::maxCutCellStabilizationScale());
}

TEST(CutIntegrationInfrastructure, BindsCutAdjacentFacetScalesFromSelectedSideMetadata)
{
    CutIntegrationContext context;

    CutCellAssemblyMetadata active_fragment;
    active_fragment.cell = 4;
    active_fragment.parent_entity = 4;
    active_fragment.volume_fraction = 0.25;
    active_fragment.side = CutIntegrationSide::Negative;

    CutCellAssemblyMetadata inactive_fragment;
    inactive_fragment.cell = 4;
    inactive_fragment.parent_entity = 4;
    inactive_fragment.volume_fraction = 1.0e-6;
    inactive_fragment.side = CutIntegrationSide::Positive;

    const auto add_rule = [&](CutCellAssemblyMetadata metadata) {
        CutQuadratureRule rule;
        rule.kind = CutQuadratureKind::Volume;
        rule.side = metadata.side;
        rule.measure = metadata.volume_fraction;
        rule.parent_measure = 1.0;
        rule.volume_fraction = metadata.volume_fraction;
        context.addGeneratedVolumeRule(133, std::move(metadata), std::move(rule));
    };
    add_rule(active_fragment);
    add_rule(inactive_fragment);

    CutFacetSetHandle handle;
    handle.marker = 133;
    handle.name = "active-side-cut-facets";
    handle.facets = {20};
    handle.facet_metadata = {
        CutFacetSetFacetMetadata{.facet = 20,
                                  .first_cell = 4,
                                  .second_cell = 5,
                                  .stabilization_scale = 0.0,
                                  .stable_id = 1u}};

    context.bindFacetStabilizationScalesForMarkerAndSide(
        handle, 133, CutIntegrationSide::Negative);

    EXPECT_DOUBLE_EQ(handle.stabilizationScaleForFacet(20), 4.0);

    CutFacetSetHandle inactive_handle = handle;
    inactive_handle.facet_metadata[0].stabilization_scale = 0.0;
    context.bindFacetStabilizationScalesForMarkerAndSide(
        inactive_handle, 133, CutIntegrationSide::Positive);
    EXPECT_DOUBLE_EQ(inactive_handle.stabilizationScaleForFacet(20),
                     CutIntegrationContext::maxCutCellStabilizationScale());
}

TEST(CutIntegrationInfrastructure, PrunesGeneratedSliverVolumeRules)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/12,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/7);
    request.interface_marker = 140;

    LevelSetInterfaceDomain domain(request);

    CutInterfaceVolumeRegion sliver;
    sliver.parent_cell = 4;
    sliver.side = CutIntegrationSide::Negative;
    sliver.parent_measure = 1.0;
    sliver.measure = CutIntegrationContext::minGeneratedCutVolumeFraction() * 0.25;
    sliver.volume_fraction =
        CutIntegrationContext::minGeneratedCutVolumeFraction() * 0.25;
    sliver.centroid = {{0.1, 0.0, 0.0}};
    sliver.normal = {{1.0, 0.0, 0.0}};
    domain.addVolumeRegion(sliver);

    CutInterfaceVolumeRegion kept;
    kept.parent_cell = 5;
    kept.side = CutIntegrationSide::Negative;
    kept.parent_measure = 1.0;
    kept.measure = 0.25;
    kept.volume_fraction = 0.25;
    kept.centroid = {{0.5, 0.0, 0.0}};
    kept.normal = {{1.0, 0.0, 0.0}};
    domain.addVolumeRegion(kept);

    CutIntegrationContext context;
    context.addGeneratedInterfaceDomain(domain);

    EXPECT_EQ(context.generatedPrunedVolumeRuleCount(), 1u);
    EXPECT_NEAR(context.generatedPrunedVolumeMeasure(), sliver.measure, 1.0e-16);
    const auto rules = context.generatedVolumeRulesForMarkerAndSide(
        140, CutIntegrationSide::Negative);
    ASSERT_EQ(rules.size(), 1u);
    EXPECT_EQ(rules.front()->provenance.parent_entity, 5);
}

TEST(CutIntegrationInfrastructure, SmallGeneratedCutFragmentsFeedConditioningNeighborhoods)
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/11,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/1);
    request.interface_marker = 55;
    request.tolerance = 1.0e-12;

    LevelSetInterfaceDomain domain(request);
    appendLinearLevelSetCellCut2D(
        domain,
        LevelSetCellCutInput{.parent_cell = 4,
                             .element_type = ElementType::Triangle3,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-1.0e-7, 1.0, 1.0}});
    ASSERT_EQ(domain.fragments().size(), 1u);
    const auto& fragment = domain.fragments().front();
    EXPECT_EQ(fragment.degeneracy, CutInterfaceDegeneracy::NearlyTangent);

    const auto diagnostic =
        diagnoseCutConditioning({fragment.measure}, 1.0e-5, 1.0e-14);
    EXPECT_TRUE(diagnostic.ok);
    EXPECT_EQ(diagnostic.small_cut_cell_count, 1u);

    const auto neighborhoods = buildCutConditioningNeighborhoods(
        domain.cutCells(),
        {fragment.measure},
        {{4, 2}, {4, 6}},
        1.0e-5);
    ASSERT_EQ(neighborhoods.size(), 1u);
    EXPECT_EQ(neighborhoods.front().cut_cell, 4);
    EXPECT_EQ(neighborhoods.front().adjacent_cells.size(), 2u);
    EXPECT_GT(neighborhoods.front().conditioning_indicator, 1.0e5);
}

TEST(CutIntegrationInfrastructure, RefinedSmallCutConditioningScalesStayBounded)
{
    constexpr int marker = 134;
    const std::vector<MeshIndex> cells{10, 11, 12, 13};
    const std::vector<Real> fractions{1.0e-2, 1.0e-4, 1.0e-8,
                                      CutIntegrationContext::minGeneratedCutVolumeFraction()};

    const auto diagnostic =
        diagnoseCutConditioning(fractions, /*small_fraction_threshold=*/1.0e-3,
                                /*degenerate_threshold=*/1.0e-13);
    EXPECT_TRUE(diagnostic.ok);
    EXPECT_EQ(diagnostic.small_cut_cell_count, 3u);
    EXPECT_EQ(diagnostic.degenerate_cut_count, 0u);

    const auto neighborhoods = buildCutConditioningNeighborhoods(
        cells,
        fractions,
        {{10, 20}, {11, 21}, {12, 22}, {13, 23}},
        /*small_fraction_threshold=*/1.0e-3);
    ASSERT_EQ(neighborhoods.size(), 3u);
    for (const auto& neighborhood : neighborhoods) {
        EXPECT_TRUE(std::isfinite(neighborhood.conditioning_indicator));
        EXPECT_GT(neighborhood.conditioning_indicator, 1.0e3);
        EXPECT_FALSE(neighborhood.adjacent_cells.empty());
    }

    CutIntegrationContext context;
    for (std::size_t i = 0; i < cells.size(); ++i) {
        CutCellAssemblyMetadata metadata;
        metadata.cell = cells[i];
        metadata.parent_entity = cells[i];
        metadata.volume_fraction = fractions[i];
        metadata.side = CutIntegrationSide::Negative;

        CutQuadratureRule rule;
        rule.kind = CutQuadratureKind::Volume;
        rule.side = CutIntegrationSide::Negative;
        rule.measure = fractions[i];
        rule.parent_measure = 1.0;
        rule.volume_fraction = fractions[i];
        context.addGeneratedVolumeRule(marker, metadata, rule);
    }

    CutCellAssemblyMetadata pruned_metadata;
    pruned_metadata.cell = 14;
    pruned_metadata.parent_entity = 14;
    pruned_metadata.volume_fraction =
        CutIntegrationContext::minGeneratedCutVolumeFraction() * 0.5;
    pruned_metadata.side = CutIntegrationSide::Negative;
    CutQuadratureRule pruned_rule;
    pruned_rule.kind = CutQuadratureKind::Volume;
    pruned_rule.side = CutIntegrationSide::Negative;
    pruned_rule.measure = pruned_metadata.volume_fraction;
    pruned_rule.parent_measure = 1.0;
    pruned_rule.volume_fraction = pruned_metadata.volume_fraction;
    context.addGeneratedVolumeRule(marker, pruned_metadata, pruned_rule);
    EXPECT_EQ(context.generatedPrunedVolumeRuleCount(), 1u);

    CutFacetSetHandle handle;
    handle.marker = marker;
    handle.name = "refined-small-cut-facets";
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const auto facet = static_cast<MeshIndex>(200 + i);
        handle.facets.push_back(facet);
        handle.facet_metadata.push_back(CutFacetSetFacetMetadata{
            .facet = facet,
            .first_cell = cells[i],
            .second_cell = static_cast<MeshIndex>(300 + i),
            .stabilization_scale = 0.0,
            .stable_id = static_cast<std::uint64_t>(i + 1)});
    }

    context.bindFacetStabilizationScalesForMarkerAndSide(
        handle, marker, CutIntegrationSide::Negative);

    for (std::size_t i = 0; i < cells.size(); ++i) {
        const auto facet = static_cast<MeshIndex>(200 + i);
        const Real expected =
            std::min(CutIntegrationContext::maxCutCellStabilizationScale(),
                     Real{1.0} / fractions[i]);
        const Real scale = handle.stabilizationScaleForFacet(facet);
        EXPECT_TRUE(std::isfinite(scale));
        EXPECT_LE(scale, CutIntegrationContext::maxCutCellStabilizationScale());
        EXPECT_DOUBLE_EQ(scale, expected);
    }
}

TEST(CutIntegrationInfrastructure, IntegratesScalarOperatorsOverCutAdjacentFacetSet)
{
    CutIntegrationContext context;

    CutQuadratureRule rule;
    rule.kind = CutQuadratureKind::Face;
    rule.side = CutIntegrationSide::Interface;
    rule.measure = 2.0;
    rule.provenance.parent_entity = 20;
    rule.provenance.predicate_policy_key = 77;
    rule.points = {
        CutQuadraturePoint{.point = {{0.0, 0.0, 0.0}},
                           .normal = {{1.0, 0.0, 0.0}},
                           .weight = 1.0},
        CutQuadraturePoint{.point = {{2.0, 0.0, 0.0}},
                           .normal = {{1.0, 0.0, 0.0}},
                           .weight = 1.0}};
    context.addFacetSetRule(/*marker=*/130, rule);

    EXPECT_TRUE(context.hasFacetSetMarker(130));
    ASSERT_EQ(context.facetSetMarkers().size(), 1u);
    EXPECT_EQ(context.facetSetMarkers().front(), 130);
    ASSERT_EQ(context.facetSetRulesForMarker(130).size(), 1u);
    EXPECT_TRUE(context.facetSetRulesForMarker(131).empty());

    const auto evaluation = context.evaluateScalarFacetSetOperator(
        /*marker=*/130,
        CutIntegrationAssemblyPath::Standard,
        [](const CutScalarOperatorPoint& point) {
            return Real{1.0} + point.point[0];
        });

    EXPECT_EQ(evaluation.marker, 130);
    EXPECT_EQ(evaluation.facet_rule_count, 1u);
    EXPECT_EQ(evaluation.facet_point_count, 2u);
    EXPECT_NEAR(evaluation.measure, 2.0, 1.0e-14);
    EXPECT_NEAR(evaluation.integral, 4.0, 1.0e-14);

    const auto empty_evaluation = context.evaluateScalarFacetSetOperator(
        /*marker=*/131,
        CutIntegrationAssemblyPath::Standard,
        [](const CutScalarOperatorPoint&) {
            return Real{1.0};
        });
    EXPECT_EQ(empty_evaluation.facet_rule_count, 0u);
    EXPECT_NEAR(empty_evaluation.integral, 0.0, 1.0e-14);
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
TEST(CutIntegrationInfrastructure, ImportsClosedCutTopologyIntoAllAssemblyPathBindings)
{
    auto mesh = make_cut_context_tetra_mesh();
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto map = svmp::search::classify_embedded_geometry(mesh, make_cut_context_plane(), options);
    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map);
    ASSERT_TRUE(svmp::search::diagnose_cut_topology_validity(topology).ok);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 123);

    ASSERT_EQ(context.volumeRules().size(), 2u);
    ASSERT_EQ(context.metadata().size(), 2u);
    ASSERT_EQ(context.bindings().size(), 2u);
    ASSERT_EQ(context.interfaceRules().size(), 1u);
    Real volume_sum = 0.0;
    for (const auto& rule : context.volumeRules()) {
        EXPECT_EQ(rule.policy.kind, CutQuadratureConstructionKind::TopologySubdivision);
        EXPECT_EQ(rule.provenance.cut_topology_revision, topology.topology_revision);
        EXPECT_FALSE(rule.points.empty());
        volume_sum += rule.measure;
    }
    EXPECT_NEAR(volume_sum, context.volumeRules()[0].parent_measure, 1.0e-10);
    for (const auto& binding : context.bindings()) {
        EXPECT_EQ(binding.visible_to_paths.size(), 6u);
        EXPECT_EQ(binding.quadrature_policy_key, 123u);
    }
}

TEST(CutIntegrationInfrastructure, ImportsCurvedIsoparametricTopologyQuadrature)
{
    auto mesh = make_cut_context_quadratic_triangle_mesh();
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto map = svmp::search::classify_embedded_geometry(
        mesh,
        make_axis_plane(0, 0.25, 222, "quadratic-triangle-plane"),
        options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    svmp::search::CutTopologyOptions topology_options;
    topology_options.allow_linearized_high_order_geometry = true;
    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map, topology_options);
    ASSERT_TRUE(topology.supported);
    ASSERT_FALSE(topology.curved_patches.empty());
    EXPECT_NE(std::find_if(topology.curved_patches.begin(),
                           topology.curved_patches.end(),
                           [](const auto& patch) {
                               return patch.isoparametric_quadrature_available &&
                                      patch.quadrature_measure > 0.0;
                           }),
              topology.curved_patches.end());

    const auto validity = svmp::search::diagnose_cut_topology_validity(
        topology,
        /*high_order_parent_geometry=*/true);
    EXPECT_TRUE(validity.ok);
    EXPECT_TRUE(validity.requires_curved_geometry_support);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 456);

    ASSERT_GE(context.volumeRules().size(), 2u);
    bool saw_curved_volume = false;
    for (const auto& rule : context.volumeRules()) {
        saw_curved_volume = saw_curved_volume ||
                            (rule.curved_geometry &&
                             rule.policy.kind == CutQuadratureConstructionKind::CurvedTopologySubdivision);
    }
    EXPECT_TRUE(saw_curved_volume);

    bool saw_curved_interface = false;
    for (const auto& rule : context.interfaceRules()) {
        saw_curved_interface = saw_curved_interface ||
                               (rule.curved_geometry &&
                                rule.policy.kind == CutQuadratureConstructionKind::CurvedTopologySubdivision &&
                                rule.measure > 0.0 &&
                                !rule.points.empty());
    }
    EXPECT_TRUE(saw_curved_interface);
}

TEST(CutIntegrationInfrastructure, ImportsTrueCurvedLineArrangementMetadata)
{
    auto mesh = make_cut_context_cubic_line_mesh();
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto embedded = make_axis_plane(0, 0.50, 225, "true-curved-line-plane");
    const auto map = svmp::search::classify_embedded_geometry(mesh, embedded, options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    svmp::search::CutTopologyOptions topology_options;
    topology_options.allow_linearized_high_order_geometry = false;
    topology_options.curved_arrangement_mode = svmp::search::CutCurvedArrangementMode::TrueArrangement;
    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map, topology_options);
    ASSERT_TRUE(topology.supported);
    ASSERT_EQ(topology.curved_patches.size(), 1u);
    EXPECT_TRUE(topology.curved_patches.front().exact_topology_available);
    EXPECT_FALSE(topology.curved_patches.front().linearized_surrogate);
    EXPECT_EQ(topology.curved_patches.front().construction_policy,
              "true-curved-isoparametric-arrangement");

    const auto validity = svmp::search::diagnose_cut_topology_validity(
        topology,
        /*high_order_parent_geometry=*/true);
    EXPECT_TRUE(validity.ok) << join_cut_validity_messages(validity.messages);
    EXPECT_FALSE(validity.requires_curved_geometry_support);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 789);
    expect_context_sensitivity_metadata(
        context, topology, 789, "true-curved-isoparametric-arrangement");

    ASSERT_EQ(context.volumeRules().size(), 2u);
    ASSERT_EQ(context.interfaceRules().size(), 1u);
    Real volume_sum = 0.0;
    Real parent_measure = -1.0;
    for (const auto& rule : context.volumeRules()) {
        EXPECT_TRUE(rule.curved_geometry);
        EXPECT_EQ(rule.policy.kind, CutQuadratureConstructionKind::CurvedTopologySubdivision);
        EXPECT_EQ(rule.policy.name, "true-curved-isoparametric-arrangement");
        ASSERT_FALSE(rule.points.empty());
        volume_sum += rule.measure;
        if (parent_measure < 0.0) {
            parent_measure = rule.parent_measure;
        } else {
            EXPECT_NEAR(rule.parent_measure,
                        parent_measure,
                        std::max(Real{1.0e-8}, parent_measure * Real{1.0e-8}));
        }
    }
    ASSERT_GT(parent_measure, 0.0);
    EXPECT_NEAR(volume_sum,
                parent_measure,
                std::max(Real{1.0e-8}, parent_measure * Real{1.0e-8}));

    const auto& interface_rule = context.interfaceRules().front();
    EXPECT_TRUE(interface_rule.curved_geometry);
    EXPECT_EQ(interface_rule.policy.kind, CutQuadratureConstructionKind::CurvedTopologySubdivision);
    EXPECT_EQ(interface_rule.policy.name, "true-curved-isoparametric-arrangement");
    ASSERT_FALSE(interface_rule.points.empty());
    EXPECT_GT(interface_rule.measure, 0.0);
    expect_curved_context_path_parity(context, Real{1.0e-8});
}

void expect_true_curved_face_context_import(
    const char* name,
    svmp::MeshBase mesh,
    const svmp::search::EmbeddedGeometryDescriptor& embedded,
    svmp::CellFamily expected_family)
{
    SCOPED_TRACE(name);
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto map = svmp::search::classify_embedded_geometry(mesh, embedded, options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    svmp::search::CutTopologyOptions topology_options;
    topology_options.allow_linearized_high_order_geometry = false;
    topology_options.curved_arrangement_mode = svmp::search::CutCurvedArrangementMode::TrueArrangement;
    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map, topology_options);
    ASSERT_TRUE(topology.supported) << (topology.diagnostics.empty() ? "" : topology.diagnostics.front());
    ASSERT_EQ(topology.curved_patches.size(), 1u);
    EXPECT_EQ(topology.curved_patches.front().parent_family, expected_family);
    EXPECT_TRUE(topology.curved_patches.front().exact_topology_available);
    EXPECT_FALSE(topology.curved_patches.front().linearized_surrogate);
    EXPECT_EQ(topology.curved_patches.front().construction_policy,
              "true-curved-isoparametric-arrangement");
    EXPECT_GT(topology.curved_patches.front().quadrature_points.size(), 2u);

    const auto validity = svmp::search::diagnose_cut_topology_validity(
        topology,
        /*high_order_parent_geometry=*/true);
    EXPECT_TRUE(validity.ok) << join_cut_validity_messages(validity.messages);
    EXPECT_FALSE(validity.requires_curved_geometry_support);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 790);
    expect_context_sensitivity_metadata(
        context, topology, 790, "true-curved-isoparametric-arrangement");

    ASSERT_EQ(context.volumeRules().size(), 2u);
    ASSERT_EQ(context.interfaceRules().size(), 1u);
    Real volume_sum = 0.0;
    Real parent_measure = -1.0;
    for (const auto& rule : context.volumeRules()) {
        EXPECT_TRUE(rule.curved_geometry);
        EXPECT_EQ(rule.policy.kind, CutQuadratureConstructionKind::CurvedTopologySubdivision);
        EXPECT_EQ(rule.policy.name, "true-curved-isoparametric-arrangement");
        ASSERT_FALSE(rule.points.empty());
        volume_sum += rule.measure;
        if (parent_measure < 0.0) {
            parent_measure = rule.parent_measure;
        } else {
            EXPECT_NEAR(rule.parent_measure,
                        parent_measure,
                        std::max(Real{1.0e-6}, parent_measure * Real{1.0e-6}));
        }
    }
    ASSERT_GT(parent_measure, 0.0);
    EXPECT_NEAR(volume_sum,
                parent_measure,
                std::max(Real{1.0e-6}, parent_measure * Real{1.0e-6}));

    const auto& interface_rule = context.interfaceRules().front();
    EXPECT_TRUE(interface_rule.curved_geometry);
    EXPECT_EQ(interface_rule.policy.kind, CutQuadratureConstructionKind::CurvedTopologySubdivision);
    EXPECT_EQ(interface_rule.policy.name, "true-curved-isoparametric-arrangement");
    ASSERT_FALSE(interface_rule.points.empty());
    EXPECT_GT(interface_rule.measure, 0.0);
    expect_curved_context_path_parity(context, Real{1.0e-6});
}

void expect_true_curved_subdivision_context_import(
    const char* name,
    svmp::MeshBase mesh,
    const svmp::search::EmbeddedGeometryDescriptor& embedded,
    svmp::CellFamily expected_family)
{
    SCOPED_TRACE(name);
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto map = svmp::search::classify_embedded_geometry(mesh, embedded, options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    svmp::search::CutTopologyOptions topology_options;
    topology_options.allow_linearized_high_order_geometry = false;
    topology_options.curved_arrangement_mode = svmp::search::CutCurvedArrangementMode::TrueArrangement;
    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map, topology_options);
    ASSERT_TRUE(topology.supported) << (topology.diagnostics.empty() ? "" : topology.diagnostics.front());
    ASSERT_GT(topology.curved_patches.size(), 1u);
    for (const auto& patch : topology.curved_patches) {
        EXPECT_EQ(patch.parent_family, expected_family);
        EXPECT_TRUE(patch.exact_topology_available);
        EXPECT_FALSE(patch.linearized_surrogate);
        EXPECT_EQ(patch.construction_policy, "true-curved-subdivision-arrangement");
        EXPECT_TRUE(patch.isoparametric_quadrature_available);
        if (embedded.kind == svmp::search::EmbeddedGeometryKind::BooleanComposite) {
            EXPECT_EQ(patch.embedded_kind, svmp::search::EmbeddedGeometryKind::BooleanComposite);
            EXPECT_EQ(patch.composition_operation, embedded.boolean_operation);
            EXPECT_EQ(patch.active_child_ordinals.size(), patch.ordered_vertices.size());
            EXPECT_EQ(patch.active_child_provenance.size(), patch.ordered_vertices.size());
        }
        EXPECT_GT(patch.quadrature_measure, 0.0);
    }

    const auto validity = svmp::search::diagnose_cut_topology_validity(
        topology,
        /*high_order_parent_geometry=*/true);
    EXPECT_TRUE(validity.ok) << join_cut_validity_messages(validity.messages);
    EXPECT_FALSE(validity.requires_curved_geometry_support);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 795);
    expect_context_sensitivity_metadata(
        context, topology, 795, "true-curved-subdivision-arrangement");

    ASSERT_EQ(context.volumeRules().size(), 2u);
    ASSERT_GT(context.interfaceRules().size(), 1u);
    Real volume_sum = 0.0;
    Real parent_measure = -1.0;
    for (const auto& rule : context.volumeRules()) {
        EXPECT_TRUE(rule.curved_geometry);
        EXPECT_EQ(rule.policy.kind, CutQuadratureConstructionKind::CurvedTopologySubdivision);
        EXPECT_EQ(rule.policy.name, "true-curved-subdivision-arrangement");
        ASSERT_FALSE(rule.points.empty());
        volume_sum += rule.measure;
        if (parent_measure < 0.0) {
            parent_measure = rule.parent_measure;
        } else {
            EXPECT_NEAR(rule.parent_measure,
                        parent_measure,
                        std::max(Real{1.0e-5}, parent_measure * Real{1.0e-5}));
        }
    }
    ASSERT_GT(parent_measure, 0.0);
    EXPECT_NEAR(volume_sum,
                parent_measure,
                std::max(Real{1.0e-4}, parent_measure * Real{1.0e-4}));

    for (const auto& interface_rule : context.interfaceRules()) {
        EXPECT_TRUE(interface_rule.curved_geometry);
        EXPECT_EQ(interface_rule.policy.kind, CutQuadratureConstructionKind::CurvedTopologySubdivision);
        EXPECT_EQ(interface_rule.policy.name, "true-curved-subdivision-arrangement");
        ASSERT_FALSE(interface_rule.points.empty());
        EXPECT_GT(interface_rule.measure, 0.0);
    }
    expect_curved_context_path_parity(context, Real{1.0e-4});
}

TEST(CutIntegrationInfrastructure, ImportsTrueCurvedFaceArrangementMetadata)
{
    expect_true_curved_face_context_import(
        "cubic-triangle",
        make_cut_context_cubic_triangle_mesh(),
        make_axis_plane(0, 0.35, 226, "true-curved-triangle-plane"),
        svmp::CellFamily::Triangle);
    expect_true_curved_face_context_import(
        "cubic-quad",
        make_cut_context_cubic_quad_mesh(),
        make_axis_plane(0, 0.50, 227, "true-curved-quad-plane"),
        svmp::CellFamily::Quad);
}

TEST(CutIntegrationInfrastructure, ImportsTrueCurvedTetraArrangementMetadata)
{
    auto mesh = make_cut_context_cubic_tetra_mesh();
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto embedded = make_axis_plane(0, 0.25, 228, "true-curved-tetra-plane");
    const auto map = svmp::search::classify_embedded_geometry(mesh, embedded, options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    svmp::search::CutTopologyOptions topology_options;
    topology_options.allow_linearized_high_order_geometry = false;
    topology_options.curved_arrangement_mode = svmp::search::CutCurvedArrangementMode::TrueArrangement;
    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map, topology_options);
    ASSERT_TRUE(topology.supported) << (topology.diagnostics.empty() ? "" : topology.diagnostics.front());
    ASSERT_EQ(topology.curved_patches.size(), 1u);
    EXPECT_EQ(topology.curved_patches.front().parent_family, svmp::CellFamily::Tetra);
    EXPECT_TRUE(topology.curved_patches.front().exact_topology_available);
    EXPECT_FALSE(topology.curved_patches.front().linearized_surrogate);
    EXPECT_EQ(topology.curved_patches.front().construction_policy,
              "true-curved-isoparametric-arrangement");
    EXPECT_GT(topology.curved_patches.front().quadrature_points.size(), 8u);

    const auto validity = svmp::search::diagnose_cut_topology_validity(
        topology,
        /*high_order_parent_geometry=*/true);
    EXPECT_TRUE(validity.ok);
    EXPECT_FALSE(validity.requires_curved_geometry_support);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 791);
    expect_context_sensitivity_metadata(
        context, topology, 791, "true-curved-isoparametric-arrangement");

    ASSERT_EQ(context.volumeRules().size(), 2u);
    ASSERT_EQ(context.interfaceRules().size(), 1u);
    Real volume_sum = 0.0;
    Real parent_measure = -1.0;
    for (const auto& rule : context.volumeRules()) {
        EXPECT_TRUE(rule.curved_geometry);
        EXPECT_EQ(rule.policy.kind, CutQuadratureConstructionKind::CurvedTopologySubdivision);
        EXPECT_EQ(rule.policy.name, "true-curved-isoparametric-arrangement");
        ASSERT_FALSE(rule.points.empty());
        volume_sum += rule.measure;
        if (parent_measure < 0.0) {
            parent_measure = rule.parent_measure;
        } else {
            EXPECT_NEAR(rule.parent_measure,
                        parent_measure,
                        std::max(Real{1.0e-6}, parent_measure * Real{1.0e-6}));
        }
    }
    ASSERT_GT(parent_measure, 0.0);
    EXPECT_NEAR(volume_sum,
                parent_measure,
                std::max(Real{1.0e-6}, parent_measure * Real{1.0e-6}));

    const auto& interface_rule = context.interfaceRules().front();
    EXPECT_TRUE(interface_rule.curved_geometry);
    EXPECT_EQ(interface_rule.policy.kind, CutQuadratureConstructionKind::CurvedTopologySubdivision);
    EXPECT_EQ(interface_rule.policy.name, "true-curved-isoparametric-arrangement");
    ASSERT_FALSE(interface_rule.points.empty());
    EXPECT_GT(interface_rule.measure, 0.0);
    expect_curved_context_path_parity(context, Real{1.0e-6});
}

TEST(CutIntegrationInfrastructure, ImportsTrueCurvedHexArrangementMetadata)
{
    auto mesh = make_cut_context_cubic_hex_mesh();
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto embedded = make_axis_plane(0, 0.42, 229, "true-curved-hex-plane");
    const auto map = svmp::search::classify_embedded_geometry(mesh, embedded, options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    svmp::search::CutTopologyOptions topology_options;
    topology_options.allow_linearized_high_order_geometry = false;
    topology_options.curved_arrangement_mode = svmp::search::CutCurvedArrangementMode::TrueArrangement;
    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map, topology_options);
    ASSERT_TRUE(topology.supported) << (topology.diagnostics.empty() ? "" : topology.diagnostics.front());
    ASSERT_EQ(topology.curved_patches.size(), 1u);
    EXPECT_EQ(topology.curved_patches.front().parent_family, svmp::CellFamily::Hex);
    EXPECT_TRUE(topology.curved_patches.front().exact_topology_available);
    EXPECT_FALSE(topology.curved_patches.front().linearized_surrogate);
    EXPECT_EQ(topology.curved_patches.front().construction_policy,
              "true-curved-isoparametric-arrangement");
    EXPECT_GT(topology.curved_patches.front().quadrature_points.size(), 8u);

    const auto validity = svmp::search::diagnose_cut_topology_validity(
        topology,
        /*high_order_parent_geometry=*/true);
    EXPECT_TRUE(validity.ok);
    EXPECT_FALSE(validity.requires_curved_geometry_support);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 792);
    expect_context_sensitivity_metadata(
        context, topology, 792, "true-curved-isoparametric-arrangement");

    ASSERT_EQ(context.volumeRules().size(), 2u);
    ASSERT_EQ(context.interfaceRules().size(), 1u);
    Real volume_sum = 0.0;
    Real parent_measure = -1.0;
    for (const auto& rule : context.volumeRules()) {
        EXPECT_TRUE(rule.curved_geometry);
        EXPECT_EQ(rule.policy.kind, CutQuadratureConstructionKind::CurvedTopologySubdivision);
        EXPECT_EQ(rule.policy.name, "true-curved-isoparametric-arrangement");
        ASSERT_FALSE(rule.points.empty());
        volume_sum += rule.measure;
        if (parent_measure < 0.0) {
            parent_measure = rule.parent_measure;
        } else {
            EXPECT_NEAR(rule.parent_measure,
                        parent_measure,
                        std::max(Real{1.0e-6}, parent_measure * Real{1.0e-6}));
        }
    }
    ASSERT_GT(parent_measure, 0.0);
    EXPECT_NEAR(volume_sum,
                parent_measure,
                std::max(Real{1.0e-6}, parent_measure * Real{1.0e-6}));

    const auto& interface_rule = context.interfaceRules().front();
    EXPECT_TRUE(interface_rule.curved_geometry);
    EXPECT_EQ(interface_rule.policy.kind, CutQuadratureConstructionKind::CurvedTopologySubdivision);
    EXPECT_EQ(interface_rule.policy.name, "true-curved-isoparametric-arrangement");
    ASSERT_FALSE(interface_rule.points.empty());
    EXPECT_GT(interface_rule.measure, 0.0);
    expect_curved_context_path_parity(context, Real{1.0e-6});
}

TEST(CutIntegrationInfrastructure, ImportsTrueCurvedWedgeArrangementMetadata)
{
    auto mesh = make_cut_context_cubic_wedge_mesh();
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto embedded = make_axis_plane(0, 0.42, 230, "true-curved-wedge-plane");
    const auto map = svmp::search::classify_embedded_geometry(mesh, embedded, options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    svmp::search::CutTopologyOptions topology_options;
    topology_options.allow_linearized_high_order_geometry = false;
    topology_options.curved_arrangement_mode = svmp::search::CutCurvedArrangementMode::TrueArrangement;
    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map, topology_options);
    ASSERT_TRUE(topology.supported) << (topology.diagnostics.empty() ? "" : topology.diagnostics.front());
    ASSERT_EQ(topology.curved_patches.size(), 1u);
    EXPECT_EQ(topology.curved_patches.front().parent_family, svmp::CellFamily::Wedge);
    EXPECT_TRUE(topology.curved_patches.front().exact_topology_available);
    EXPECT_FALSE(topology.curved_patches.front().linearized_surrogate);
    EXPECT_EQ(topology.curved_patches.front().construction_policy,
              "true-curved-isoparametric-arrangement");
    EXPECT_GT(topology.curved_patches.front().quadrature_points.size(), 8u);

    const auto validity = svmp::search::diagnose_cut_topology_validity(
        topology,
        /*high_order_parent_geometry=*/true);
    EXPECT_TRUE(validity.ok);
    EXPECT_FALSE(validity.requires_curved_geometry_support);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 793);
    expect_context_sensitivity_metadata(
        context, topology, 793, "true-curved-isoparametric-arrangement");

    ASSERT_EQ(context.volumeRules().size(), 2u);
    ASSERT_EQ(context.interfaceRules().size(), 1u);
    Real volume_sum = 0.0;
    Real parent_measure = -1.0;
    for (const auto& rule : context.volumeRules()) {
        EXPECT_TRUE(rule.curved_geometry);
        EXPECT_EQ(rule.policy.kind, CutQuadratureConstructionKind::CurvedTopologySubdivision);
        EXPECT_EQ(rule.policy.name, "true-curved-isoparametric-arrangement");
        ASSERT_FALSE(rule.points.empty());
        volume_sum += rule.measure;
        if (parent_measure < 0.0) {
            parent_measure = rule.parent_measure;
        } else {
            EXPECT_NEAR(rule.parent_measure,
                        parent_measure,
                        std::max(Real{1.0e-6}, parent_measure * Real{1.0e-6}));
        }
    }
    ASSERT_GT(parent_measure, 0.0);
    EXPECT_NEAR(volume_sum,
                parent_measure,
                std::max(Real{1.0e-6}, parent_measure * Real{1.0e-6}));

    const auto& interface_rule = context.interfaceRules().front();
    EXPECT_TRUE(interface_rule.curved_geometry);
    EXPECT_EQ(interface_rule.policy.kind, CutQuadratureConstructionKind::CurvedTopologySubdivision);
    EXPECT_EQ(interface_rule.policy.name, "true-curved-isoparametric-arrangement");
    ASSERT_FALSE(interface_rule.points.empty());
    EXPECT_GT(interface_rule.measure, 0.0);
    expect_curved_context_path_parity(context, Real{1.0e-6});
}

TEST(CutIntegrationInfrastructure, ImportsTrueCurvedPyramidArrangementMetadata)
{
    auto mesh = make_cut_context_cubic_pyramid_mesh();
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto embedded = make_axis_plane(0, 0.42, 231, "true-curved-pyramid-plane");
    const auto map = svmp::search::classify_embedded_geometry(mesh, embedded, options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    svmp::search::CutTopologyOptions topology_options;
    topology_options.allow_linearized_high_order_geometry = false;
    topology_options.curved_arrangement_mode = svmp::search::CutCurvedArrangementMode::TrueArrangement;
    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map, topology_options);
    ASSERT_TRUE(topology.supported) << (topology.diagnostics.empty() ? "" : topology.diagnostics.front());
    ASSERT_EQ(topology.curved_patches.size(), 1u);
    EXPECT_EQ(topology.curved_patches.front().parent_family, svmp::CellFamily::Pyramid);
    EXPECT_TRUE(topology.curved_patches.front().exact_topology_available);
    EXPECT_FALSE(topology.curved_patches.front().linearized_surrogate);
    EXPECT_EQ(topology.curved_patches.front().construction_policy,
              "true-curved-isoparametric-arrangement");
    EXPECT_GT(topology.curved_patches.front().quadrature_points.size(), 8u);

    const auto validity = svmp::search::diagnose_cut_topology_validity(
        topology,
        /*high_order_parent_geometry=*/true);
    EXPECT_TRUE(validity.ok);
    EXPECT_FALSE(validity.requires_curved_geometry_support);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 794);
    expect_context_sensitivity_metadata(
        context, topology, 794, "true-curved-isoparametric-arrangement");

    ASSERT_EQ(context.volumeRules().size(), 2u);
    ASSERT_EQ(context.interfaceRules().size(), 1u);
    Real volume_sum = 0.0;
    Real parent_measure = -1.0;
    for (const auto& rule : context.volumeRules()) {
        EXPECT_TRUE(rule.curved_geometry);
        EXPECT_EQ(rule.policy.kind, CutQuadratureConstructionKind::CurvedTopologySubdivision);
        EXPECT_EQ(rule.policy.name, "true-curved-isoparametric-arrangement");
        ASSERT_FALSE(rule.points.empty());
        volume_sum += rule.measure;
        if (parent_measure < 0.0) {
            parent_measure = rule.parent_measure;
        } else {
            EXPECT_NEAR(rule.parent_measure,
                        parent_measure,
                        std::max(Real{1.0e-6}, parent_measure * Real{1.0e-6}));
        }
    }
    ASSERT_GT(parent_measure, 0.0);
    EXPECT_NEAR(volume_sum,
                parent_measure,
                std::max(Real{1.0e-6}, parent_measure * Real{1.0e-6}));

    const auto& interface_rule = context.interfaceRules().front();
    EXPECT_TRUE(interface_rule.curved_geometry);
    EXPECT_EQ(interface_rule.policy.kind, CutQuadratureConstructionKind::CurvedTopologySubdivision);
    EXPECT_EQ(interface_rule.policy.name, "true-curved-isoparametric-arrangement");
    ASSERT_FALSE(interface_rule.points.empty());
    EXPECT_GT(interface_rule.measure, 0.0);
    expect_curved_context_path_parity(context, Real{1.0e-6});
}

TEST(CutIntegrationInfrastructure, ImportsTrueCurvedNonGraphSubdivisionArrangementMetadata)
{
    expect_true_curved_subdivision_context_import(
        "non-graph-cubic-triangle",
        make_cut_context_cubic_non_graph_triangle_mesh(),
        make_axis_plane(0, 0.04, 230, "true-curved-non-graph-triangle-plane"),
        svmp::CellFamily::Triangle);
    expect_true_curved_subdivision_context_import(
        "non-graph-cubic-quad",
        make_cut_context_cubic_non_graph_quad_mesh(),
        make_axis_plane(0, 0.25, 231, "true-curved-non-graph-quad-plane"),
        svmp::CellFamily::Quad);
    expect_true_curved_subdivision_context_import(
        "non-graph-cubic-hex",
        make_cut_context_cubic_non_graph_hex_mesh(),
        make_axis_plane(0, 0.25, 232, "true-curved-non-graph-hex-plane"),
        svmp::CellFamily::Hex);
    expect_true_curved_subdivision_context_import(
        "non-graph-cubic-wedge",
        make_cut_context_cubic_non_graph_wedge_mesh(),
        make_axis_plane(0, 0.25, 233, "true-curved-non-graph-wedge-plane"),
        svmp::CellFamily::Wedge);
    expect_true_curved_subdivision_context_import(
        "non-graph-cubic-pyramid",
        make_cut_context_cubic_non_graph_pyramid_mesh(),
        make_axis_plane(0, 0.04, 234, "true-curved-non-graph-pyramid-plane"),
        svmp::CellFamily::Pyramid);
}

TEST(CutIntegrationInfrastructure, ImportsTrueCurvedNonPlaneSubdivisionArrangementMetadata)
{
    const auto check_all_families =
        [&](const std::string& prefix,
            std::uint64_t base_epoch,
            const auto& make_line,
            const auto& make_triangle,
            const auto& make_quad,
            const auto& make_tetra,
            const auto& make_hex,
            const auto& make_wedge,
            const auto& make_pyramid) {
            expect_true_curved_subdivision_context_import(
                (prefix + "-line").c_str(),
                make_cut_context_cubic_line_mesh(),
                make_line(base_epoch + 1, prefix + "-line"),
                svmp::CellFamily::Line);
            expect_true_curved_subdivision_context_import(
                (prefix + "-triangle").c_str(),
                make_cut_context_cubic_triangle_mesh(),
                make_triangle(base_epoch + 2, prefix + "-triangle"),
                svmp::CellFamily::Triangle);
            expect_true_curved_subdivision_context_import(
                (prefix + "-quad").c_str(),
                make_cut_context_cubic_quad_mesh(),
                make_quad(base_epoch + 3, prefix + "-quad"),
                svmp::CellFamily::Quad);
            expect_true_curved_subdivision_context_import(
                (prefix + "-tetra").c_str(),
                make_cut_context_cubic_tetra_mesh(),
                make_tetra(base_epoch + 4, prefix + "-tetra"),
                svmp::CellFamily::Tetra);
            expect_true_curved_subdivision_context_import(
                (prefix + "-hex").c_str(),
                make_cut_context_cubic_hex_mesh(),
                make_hex(base_epoch + 5, prefix + "-hex"),
                svmp::CellFamily::Hex);
            expect_true_curved_subdivision_context_import(
                (prefix + "-wedge").c_str(),
                make_cut_context_cubic_wedge_mesh(),
                make_wedge(base_epoch + 6, prefix + "-wedge"),
                svmp::CellFamily::Wedge);
            expect_true_curved_subdivision_context_import(
                (prefix + "-pyramid").c_str(),
                make_cut_context_cubic_pyramid_mesh(),
                make_pyramid(base_epoch + 7, prefix + "-pyramid"),
                svmp::CellFamily::Pyramid);
        };

    check_all_families(
        "sphere",
        2350,
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_sphere({{0.48, 0.0, 0.0}}, 0.22, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_sphere({{0.35, 0.35, 0.0}}, 0.30, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_sphere({{0.50, 0.0, 0.0}}, 0.62, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_sphere({{0.32, 0.28, 0.22}}, 0.34, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_sphere({{0.50, 0.0, 0.0}}, 0.72, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_sphere({{0.50, 0.35, 0.25}}, 0.46, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_sphere({{0.48, 0.0, 0.0}}, 0.42, epoch, id);
        });

    check_all_families(
        "callback",
        2360,
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_callback_sphere({{0.48, 0.0, 0.0}}, 0.22, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_callback_sphere({{0.35, 0.35, 0.0}}, 0.30, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_callback_sphere({{0.50, 0.0, 0.0}}, 0.62, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_callback_sphere({{0.32, 0.28, 0.22}}, 0.34, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_callback_sphere({{0.50, 0.0, 0.0}}, 0.72, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_callback_sphere({{0.50, 0.35, 0.25}}, 0.46, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_callback_sphere({{0.48, 0.0, 0.0}}, 0.42, epoch, id);
        });

    check_all_families(
        "level-set",
        2370,
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_level_set_sphere({{0.48, 0.0, 0.0}}, 0.22, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_level_set_sphere({{0.35, 0.35, 0.0}}, 0.30, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_level_set_sphere({{0.50, 0.0, 0.0}}, 0.62, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_level_set_sphere({{0.32, 0.28, 0.22}}, 0.34, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_level_set_sphere({{0.50, 0.0, 0.0}}, 0.72, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_level_set_sphere({{0.50, 0.35, 0.25}}, 0.46, epoch, id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_level_set_sphere({{0.48, 0.0, 0.0}}, 0.42, epoch, id);
        });

    check_all_families(
        "triangulated",
        2380,
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_triangulated_octahedron_sphere({{0.48, 0.0, 0.0}},
                                                              0.30,
                                                              epoch,
                                                              id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_triangulated_octahedron_sphere({{0.35, 0.35, 0.0}},
                                                              0.30,
                                                              epoch,
                                                              id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_triangulated_octahedron_sphere({{0.50, 0.0, 0.0}},
                                                              0.62,
                                                              epoch,
                                                              id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_triangulated_octahedron_sphere({{0.32, 0.28, 0.22}},
                                                              0.34,
                                                              epoch,
                                                              id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_triangulated_octahedron_sphere({{0.50, 0.0, 0.0}},
                                                              0.95,
                                                              epoch,
                                                              id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_triangulated_octahedron_sphere({{0.50, 0.35, 0.25}},
                                                              0.62,
                                                              epoch,
                                                              id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_triangulated_octahedron_sphere({{0.48, 0.0, 0.0}},
                                                              0.58,
                                                              epoch,
                                                              id);
        });

    check_all_families(
        "boolean",
        2390,
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_boolean_composite(
                svmp::search::EmbeddedGeometryBooleanOperation::Intersection,
                {make_context_sphere({{0.48, 0.0, 0.0}}, 0.22, epoch + 1000, id + "-active"),
                 make_context_sphere({{0.48, 0.0, 0.0}}, 10.0, epoch + 1001, id + "-enclosing")},
                epoch,
                id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_boolean_composite(
                svmp::search::EmbeddedGeometryBooleanOperation::Intersection,
                {make_context_sphere({{0.35, 0.35, 0.0}}, 0.30, epoch + 1000, id + "-active"),
                 make_context_sphere({{0.35, 0.35, 0.0}}, 10.0, epoch + 1001, id + "-enclosing")},
                epoch,
                id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_boolean_composite(
                svmp::search::EmbeddedGeometryBooleanOperation::Intersection,
                {make_context_sphere({{0.50, 0.0, 0.0}}, 0.62, epoch + 1000, id + "-active"),
                 make_context_sphere({{0.50, 0.0, 0.0}}, 10.0, epoch + 1001, id + "-enclosing")},
                epoch,
                id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_boolean_composite(
                svmp::search::EmbeddedGeometryBooleanOperation::Intersection,
                {make_context_sphere({{0.32, 0.28, 0.22}}, 0.34, epoch + 1000, id + "-active"),
                 make_context_sphere({{0.32, 0.28, 0.22}}, 10.0, epoch + 1001, id + "-enclosing")},
                epoch,
                id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_boolean_composite(
                svmp::search::EmbeddedGeometryBooleanOperation::Union,
                {make_context_sphere({{0.50, 0.0, 0.0}}, 0.72, epoch + 1000, id + "-active"),
                 make_context_sphere({{4.0, 4.0, 4.0}}, 0.25, epoch + 1001, id + "-far")},
                epoch,
                id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_boolean_composite(
                svmp::search::EmbeddedGeometryBooleanOperation::Difference,
                {make_context_sphere({{0.50, 0.35, 0.25}}, 10.0, epoch + 1000, id + "-base"),
                 make_context_sphere({{0.50, 0.35, 0.25}}, 0.46, epoch + 1001, id + "-cavity")},
                epoch,
                id);
        },
        [](std::uint64_t epoch, const std::string& id) {
            return make_context_boolean_composite(
                svmp::search::EmbeddedGeometryBooleanOperation::Intersection,
                {make_context_sphere({{0.48, 0.0, 0.0}}, 0.42, epoch + 1000, id + "-active"),
                 make_context_sphere({{0.48, 0.0, 0.0}}, 10.0, epoch + 1001, id + "-enclosing")},
                epoch,
                id);
        });
}

TEST(CutIntegrationInfrastructure, ImportsTrueCurvedBooleanChildChildSubdivisionArrangementMetadata)
{
    auto mesh = make_cut_context_cubic_tetra_mesh();
    const auto embedded = make_context_boolean_composite(
        svmp::search::EmbeddedGeometryBooleanOperation::Intersection,
        {make_context_sphere({{0.25, 0.28, 0.22}},
                             0.34,
                             2401,
                             "true-curved-boolean-child-child-a"),
         make_context_sphere({{0.45, 0.28, 0.22}},
                             0.34,
                             2402,
                             "true-curved-boolean-child-child-b")},
        2403,
        "true-curved-boolean-child-child-intersection");

    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto map = svmp::search::classify_embedded_geometry(mesh, embedded, options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    svmp::search::CutTopologyOptions topology_options;
    topology_options.allow_linearized_high_order_geometry = false;
    topology_options.curved_arrangement_mode = svmp::search::CutCurvedArrangementMode::TrueArrangement;
    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map, topology_options);
    ASSERT_TRUE(topology.supported) << (topology.diagnostics.empty() ? "" : topology.diagnostics.front());

    bool saw_child_zero = false;
    bool saw_child_one = false;
    bool saw_mixed_child_patch = false;
    for (const auto& patch : topology.curved_patches) {
        ASSERT_EQ(patch.embedded_kind, svmp::search::EmbeddedGeometryKind::BooleanComposite);
        ASSERT_EQ(patch.active_child_ordinals.size(), patch.ordered_vertices.size());
        std::set<std::size_t> patch_children;
        for (const auto ordinal : patch.active_child_ordinals) {
            patch_children.insert(ordinal);
        }
        saw_child_zero = saw_child_zero || patch_children.count(0u) != 0u;
        saw_child_one = saw_child_one || patch_children.count(1u) != 0u;
        saw_mixed_child_patch = saw_mixed_child_patch || patch_children.size() > 1u;
    }
    EXPECT_TRUE(saw_child_zero);
    EXPECT_TRUE(saw_child_one);
    EXPECT_TRUE(saw_mixed_child_patch);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 796);
    expect_context_sensitivity_metadata(
        context, topology, 796, "true-curved-subdivision-arrangement");

    ASSERT_EQ(context.volumeRules().size(), 2u);
    ASSERT_GT(context.interfaceRules().size(), 1u);
    Real volume_sum = 0.0;
    Real parent_measure = -1.0;
    for (const auto& rule : context.volumeRules()) {
        EXPECT_TRUE(rule.curved_geometry);
        EXPECT_EQ(rule.policy.kind, CutQuadratureConstructionKind::CurvedTopologySubdivision);
        EXPECT_EQ(rule.policy.name, "true-curved-subdivision-arrangement");
        volume_sum += rule.measure;
        if (parent_measure < 0.0) {
            parent_measure = rule.parent_measure;
        } else {
            EXPECT_NEAR(rule.parent_measure,
                        parent_measure,
                        std::max(Real{1.0e-5}, parent_measure * Real{1.0e-5}));
        }
    }
    ASSERT_GT(parent_measure, 0.0);
    EXPECT_NEAR(volume_sum,
                parent_measure,
                std::max(Real{1.0e-4}, parent_measure * Real{1.0e-4}));
    expect_curved_context_path_parity(context, Real{1.0e-4});
}

TEST(CutIntegrationInfrastructure, ImportsCurvedQuadAndHexIsoparametricTopologyQuadrature)
{
    expect_curved_isoparametric_context_import(
        "quadratic-quad",
        make_cut_context_quadratic_quad_mesh(),
        make_axis_plane(0, 0.0, 223, "quadratic-quad-plane"));
    expect_curved_isoparametric_context_import(
        "quadratic-hex",
        make_cut_context_quadratic_hex_mesh(),
        make_axis_plane(0, 0.0, 224, "quadratic-hex-plane"));
}

TEST(CutIntegrationInfrastructure, QuadraticCurvedPlaneCutDataIsIdenticalAcrossAllExecutionPaths)
{
    expect_quadratic_curved_plane_context_path_parity(
        "quadratic-line",
        make_cut_context_quadratic_line_mesh(),
        make_axis_plane(0, 0.4, 330, "path-parity-quadratic-line-plane"));
    expect_quadratic_curved_plane_context_path_parity(
        "quadratic-triangle",
        make_cut_context_quadratic_triangle_mesh(),
        make_axis_plane(0, 0.25, 331, "path-parity-quadratic-triangle-plane"));
    expect_quadratic_curved_plane_context_path_parity(
        "quadratic-quad",
        make_cut_context_quadratic_quad_mesh(),
        make_axis_plane(0, 0.0, 332, "path-parity-quadratic-quad-plane"));
    expect_quadratic_curved_plane_context_path_parity(
        "quadratic-tetra",
        make_cut_context_quadratic_tetra_mesh(),
        make_axis_plane(0, 0.25, 333, "path-parity-quadratic-tetra-plane"));
    expect_quadratic_curved_plane_context_path_parity(
        "quadratic-hex",
        make_cut_context_quadratic_hex_mesh(),
        make_axis_plane(0, 0.0, 334, "path-parity-quadratic-hex-plane"));
}

TEST(CutIntegrationInfrastructure, QuadraticCurvedPlaneCutSensitivityMetadataIsVisibleAcrossAllExecutionPaths)
{
    expect_quadratic_curved_plane_sensitivity_path_visibility(
        "quadratic-line",
        make_cut_context_quadratic_line_mesh(),
        make_axis_plane(0, 0.4, 430, "sensitivity-quadratic-line-plane"));
    expect_quadratic_curved_plane_sensitivity_path_visibility(
        "quadratic-triangle",
        make_cut_context_quadratic_triangle_mesh(),
        make_axis_plane(0, 0.25, 431, "sensitivity-quadratic-triangle-plane"));
    expect_quadratic_curved_plane_sensitivity_path_visibility(
        "quadratic-quad",
        make_cut_context_quadratic_quad_mesh(),
        make_axis_plane(0, 0.0, 432, "sensitivity-quadratic-quad-plane"));
    expect_quadratic_curved_plane_sensitivity_path_visibility(
        "quadratic-tetra",
        make_cut_context_quadratic_tetra_mesh(),
        make_axis_plane(0, 0.25, 433, "sensitivity-quadratic-tetra-plane"));
    expect_quadratic_curved_plane_sensitivity_path_visibility(
        "quadratic-hex",
        make_cut_context_quadratic_hex_mesh(),
        make_axis_plane(0, 0.0, 434, "sensitivity-quadratic-hex-plane"));
}

TEST(CutIntegrationInfrastructure, ImportsLinearizedCutQuadratureForEveryAdvertisedFamily)
{
    expect_cut_context_family_quadrature(
        "line",
        make_cut_context_line_mesh(),
        make_axis_plane(0, 0.4, 21, "line-plane"),
        1.0,
        0.4,
        0.6,
        1.0);
    expect_cut_context_family_quadrature(
        "triangle",
        make_cut_context_triangle_mesh(),
        make_axis_plane(0, 0.25, 22, "triangle-plane"),
        0.5,
        0.21875,
        0.28125,
        0.75);
    expect_cut_context_family_quadrature(
        "quad",
        make_cut_context_quad_mesh(),
        make_axis_plane(0, 0.5, 23, "quad-plane"),
        1.0,
        0.5,
        0.5,
        1.0);
    expect_cut_context_family_quadrature(
        "tetra",
        make_cut_context_tetra_mesh(),
        make_axis_plane(0, 0.25, 24, "tetra-plane"),
        1.0 / 6.0,
        (1.0 - std::pow(0.75, 3.0)) / 6.0,
        std::pow(0.75, 3.0) / 6.0,
        0.5 * 0.75 * 0.75);
    expect_cut_context_family_quadrature(
        "hex",
        make_cut_context_hex_mesh(),
        make_axis_plane(0, 0.5, 25, "hex-plane"),
        1.0,
        0.5,
        0.5,
        1.0);
    expect_cut_context_family_quadrature(
        "wedge",
        make_cut_context_wedge_mesh(),
        make_axis_plane(2, 0.5, 26, "wedge-plane"),
        0.5,
        0.25,
        0.25,
        0.5);
    expect_cut_context_family_quadrature(
        "pyramid",
        make_cut_context_pyramid_mesh(),
        make_axis_plane(2, 0.4, 27, "pyramid-plane"),
        1.0 / 3.0,
        (1.0 - std::pow(0.6, 3.0)) / 3.0,
        std::pow(0.6, 3.0) / 3.0,
        0.6 * 0.6);
    expect_cut_context_family_quadrature(
        "polygon",
        make_cut_context_polygon_mesh(),
        make_axis_plane(0, 0.5, 28, "polygon-plane"),
        1.14,
        0.57,
        0.57,
        1.2);
    expect_cut_context_family_quadrature(
        "polyhedron",
        make_cut_context_polyhedron_cube_mesh(),
        make_axis_plane(0, 0.5, 29, "polyhedron-plane"),
        1.0,
        0.5,
        0.5,
        1.0);
}

TEST(CutIntegrationInfrastructure, LinearizedPlaneCutDataIsIdenticalAcrossAllExecutionPaths)
{
    expect_linearized_plane_context_path_parity(
        "line",
        make_cut_context_line_mesh(),
        make_axis_plane(0, 0.4, 319, "path-parity-line-plane"));
    expect_linearized_plane_context_path_parity(
        "triangle",
        make_cut_context_triangle_mesh(),
        make_axis_plane(0, 0.25, 320, "path-parity-triangle-plane"));
    expect_linearized_plane_context_path_parity(
        "quad",
        make_cut_context_quad_mesh(),
        make_axis_plane(0, 0.5, 321, "path-parity-quad-plane"));
    expect_linearized_plane_context_path_parity(
        "tetra",
        make_cut_context_tetra_mesh(),
        make_axis_plane(0, 0.25, 322, "path-parity-tetra-plane"));
    expect_linearized_plane_context_path_parity(
        "hex",
        make_cut_context_hex_mesh(),
        make_axis_plane(0, 0.5, 323, "path-parity-hex-plane"));
    expect_linearized_plane_context_path_parity(
        "wedge",
        make_cut_context_wedge_mesh(),
        make_axis_plane(2, 0.5, 324, "path-parity-wedge-plane"));
    expect_linearized_plane_context_path_parity(
        "pyramid",
        make_cut_context_pyramid_mesh(),
        make_axis_plane(2, 0.4, 325, "path-parity-pyramid-plane"));
    expect_linearized_plane_context_path_parity(
        "polygon",
        make_cut_context_polygon_mesh(),
        make_axis_plane(0, 0.5, 326, "path-parity-polygon-plane"));
    expect_linearized_plane_context_path_parity(
        "polyhedron",
        make_cut_context_polyhedron_cube_mesh(),
        make_axis_plane(0, 0.5, 327, "path-parity-polyhedron-plane"));
}

TEST(CutIntegrationInfrastructure, LinearizedCutDataIsIdenticalAcrossAllExecutionPaths)
{
    const std::array<svmp::CellFamily, 9> families{{
        svmp::CellFamily::Line,
        svmp::CellFamily::Triangle,
        svmp::CellFamily::Quad,
        svmp::CellFamily::Tetra,
        svmp::CellFamily::Hex,
        svmp::CellFamily::Wedge,
        svmp::CellFamily::Pyramid,
        svmp::CellFamily::Polygon,
        svmp::CellFamily::Polyhedron}};
    const std::array<svmp::search::EmbeddedGeometryKind, 4> kinds{{
        svmp::search::EmbeddedGeometryKind::Sphere,
        svmp::search::EmbeddedGeometryKind::TriangulatedSurface,
        svmp::search::EmbeddedGeometryKind::LevelSetField,
        svmp::search::EmbeddedGeometryKind::BooleanComposite}};

    std::uint64_t epoch = 500;
    for (const auto family : families) {
        for (const auto kind : kinds) {
            const std::string name =
                "linearized-nonplane-path-parity-" +
                std::to_string(static_cast<int>(family)) + "-" +
                std::to_string(static_cast<int>(kind));
            expect_linearized_context_path_parity(
                name.c_str(),
                make_first_order_cut_context_mesh(family),
                make_first_order_linearized_context_embedded(kind, epoch++, name));
        }
    }
}

TEST(CutIntegrationInfrastructure, ProductionCutScalarOperatorsMatchAnalyticHalfCubePlaneCut)
{
    auto mesh = make_cut_context_hex_mesh();
    svmp::search::CutClassificationOptions options;
    options.classify_faces = false;
    options.classify_edges = false;
    const auto embedded =
        make_axis_plane(0, 0.5, 9050, "production-operator-half-cube-plane");
    const auto map = svmp::search::classify_embedded_geometry(mesh, embedded, options);
    ASSERT_EQ(map.cells.size(), 1u);
    ASSERT_EQ(map.cells[0].classification, svmp::search::CutClassification::Cut);

    const auto topology = svmp::search::reconstruct_cut_topology(mesh, map);
    ASSERT_TRUE(topology.supported) << join_cut_validity_messages(topology.diagnostics);
    const auto validity = svmp::search::diagnose_cut_topology_validity(topology);
    ASSERT_TRUE(validity.ok) << join_cut_validity_messages(validity.messages);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 9051);

    const auto constant = context.evaluateScalarCutOperator(
        CutIntegrationAssemblyPath::Standard,
        constant_cut_operator_integrand,
        constant_cut_operator_integrand);
    EXPECT_EQ(constant.volume_rule_count, 2u);
    EXPECT_EQ(constant.interface_rule_count, 1u);
    EXPECT_NEAR(constant.parent_measure, 1.0, 1.0e-12);
    EXPECT_NEAR(constant.negative_volume_integral, 0.5, 1.0e-12);
    EXPECT_NEAR(constant.positive_volume_integral, 0.5, 1.0e-12);
    EXPECT_NEAR(constant.volumeIntegral(), 1.0, 1.0e-12);
    EXPECT_NEAR(constant.interface_integral, 1.0, 1.0e-12);
    EXPECT_NEAR(constant.interface_measure, 1.0, 1.0e-12);

    const auto x_moment = context.evaluateScalarCutOperator(
        CutIntegrationAssemblyPath::Standard,
        x_coordinate_cut_operator_integrand,
        x_coordinate_cut_operator_integrand);
    EXPECT_GT(x_moment.negative_volume_integral, 0.0);
    EXPECT_GT(x_moment.positive_volume_integral, x_moment.negative_volume_integral);
    EXPECT_NEAR(x_moment.volumeIntegral(), 0.5, 1.0e-12);
    EXPECT_NEAR(x_moment.interface_integral, 0.5, 1.0e-12);

    for (const auto path : all_cut_execution_paths()) {
        const auto actual_constant = context.evaluateScalarCutOperator(
            path, constant_cut_operator_integrand, constant_cut_operator_integrand);
        expect_equivalent_operator_evaluation(constant, actual_constant, 1.0e-12);

        const auto actual_x_moment = context.evaluateScalarCutOperator(
            path, x_coordinate_cut_operator_integrand, x_coordinate_cut_operator_integrand);
        expect_equivalent_operator_evaluation(x_moment, actual_x_moment, 1.0e-12);
    }
}

TEST(CutIntegrationInfrastructure, ProductionCutScalarOperatorsConsumeImportedTopologyAcrossAllExecutionPaths)
{
    const std::array<svmp::CellFamily, 9> families{{
        svmp::CellFamily::Line,
        svmp::CellFamily::Triangle,
        svmp::CellFamily::Quad,
        svmp::CellFamily::Tetra,
        svmp::CellFamily::Hex,
        svmp::CellFamily::Wedge,
        svmp::CellFamily::Pyramid,
        svmp::CellFamily::Polygon,
        svmp::CellFamily::Polyhedron}};
    const std::array<svmp::search::EmbeddedGeometryKind, 4> kinds{{
        svmp::search::EmbeddedGeometryKind::Sphere,
        svmp::search::EmbeddedGeometryKind::TriangulatedSurface,
        svmp::search::EmbeddedGeometryKind::LevelSetField,
        svmp::search::EmbeddedGeometryKind::BooleanComposite}};

    std::uint64_t epoch = 9000;
    for (const auto family : families) {
        const std::string plane_name =
            "production-operator-plane-" +
            std::to_string(static_cast<int>(family));
        expect_production_cut_operator_path_parity(
            plane_name.c_str(),
            make_first_order_cut_context_mesh(family),
            make_axis_plane(family == svmp::CellFamily::Wedge ||
                                family == svmp::CellFamily::Pyramid
                                ? 2
                                : 0,
                            family == svmp::CellFamily::Line ? 0.4 : 0.5,
                            epoch++,
                            plane_name),
            1.0e-10);

        for (const auto kind : kinds) {
            const std::string name =
                "production-operator-nonplane-" +
                std::to_string(static_cast<int>(family)) + "-" +
                std::to_string(static_cast<int>(kind));
            expect_production_cut_operator_path_parity(
                name.c_str(),
                make_first_order_cut_context_mesh(family),
                make_first_order_linearized_context_embedded(kind, epoch++, name),
                1.0e-8);
        }
    }
}

TEST(CutIntegrationInfrastructure, LinearizedNonPlaneCutSensitivityMetadataIsVisibleAcrossAllExecutionPaths)
{
    const std::array<svmp::CellFamily, 9> families{{
        svmp::CellFamily::Line,
        svmp::CellFamily::Triangle,
        svmp::CellFamily::Quad,
        svmp::CellFamily::Tetra,
        svmp::CellFamily::Hex,
        svmp::CellFamily::Wedge,
        svmp::CellFamily::Pyramid,
        svmp::CellFamily::Polygon,
        svmp::CellFamily::Polyhedron}};
    const std::array<svmp::search::EmbeddedGeometryKind, 4> kinds{{
        svmp::search::EmbeddedGeometryKind::Sphere,
        svmp::search::EmbeddedGeometryKind::TriangulatedSurface,
        svmp::search::EmbeddedGeometryKind::LevelSetField,
        svmp::search::EmbeddedGeometryKind::BooleanComposite}};

    std::uint64_t epoch = 640;
    for (const auto family : families) {
        for (const auto kind : kinds) {
            const std::string name =
                "linearized-nonplane-sensitivity-" +
                std::to_string(static_cast<int>(family)) + "-" +
                std::to_string(static_cast<int>(kind));
            const auto embedded =
                make_first_order_linearized_context_embedded(kind, epoch++, name);
            const auto quadrature_policy_key = 990u + epoch;
            expect_linearized_sensitivity_path_visibility(
                name.c_str(),
                make_first_order_cut_context_mesh(family),
                embedded,
                quadrature_policy_key);
        }
    }
}

TEST(CutIntegrationInfrastructure, SupportMatrixQualificationHarnessCoversFEExecutionPaths)
{
    const auto records = svmp::search::qualify_cut_support_matrix();
    ASSERT_FALSE(records.empty());

    std::set<std::string> observed_paths;
    for (const auto& record : records) {
        if (record.entry.status == svmp::search::CutSupportStatus::Unsupported) {
            continue;
        }
        EXPECT_TRUE(record.requires_fe_execution_evidence);
        EXPECT_TRUE(record.fe_execution_evidence) << record.entry.fe_execution_path;
        observed_paths.insert(record.entry.fe_execution_path);
    }

    const std::set<std::string> expected_paths{
        "standard-assembly",
        "matrix-free",
        "forms-interpreter",
        "ad",
        "symbolic-tangent",
        "jit"};
    EXPECT_EQ(observed_paths, expected_paths);
}
#endif

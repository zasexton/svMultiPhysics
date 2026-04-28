#include "Assembly/CutIntegrationContext.h"
#include "Geometry/CutQuadrature.h"
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
#include <string>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::assembly;
using namespace svmp::FE::geometry;
using namespace svmp::FE::systems;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
namespace {

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
    EXPECT_TRUE(validity.ok);

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

} // namespace
#endif

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
    EXPECT_TRUE(geometry_decision.rebuild_matrix_free_data);

    current = cached;
    current.cut_topology_revision = 77;
    auto topology_decision = classifyCutIntegrationRefresh(cached, current);
    EXPECT_TRUE(topology_decision.rebuild_quadrature);
    EXPECT_TRUE(topology_decision.update_stabilization_hooks);

    current = cached;
    current.embedded_field_value_revision = 9;
    auto field_decision = classifyCutIntegrationRefresh(cached, current);
    EXPECT_TRUE(field_decision.rebuild_cut_classification);

    current = cached;
    current.conditioning_revision = 4;
    auto conditioning_decision = classifyCutIntegrationRefresh(cached, current);
    EXPECT_TRUE(conditioning_decision.update_stabilization_hooks);
    EXPECT_TRUE(conditioning_decision.refresh_preconditioner);

    current = cached;
    current.fe_dof_layout_revision = 3;
    auto layout_decision = classifyCutIntegrationRefresh(cached, current);
    EXPECT_FALSE(layout_decision.rebuild_cut_classification);
    EXPECT_FALSE(layout_decision.rebuild_quadrature);
    EXPECT_TRUE(layout_decision.rebuild_matrix);
    EXPECT_TRUE(layout_decision.refresh_preconditioner);
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
    EXPECT_TRUE(validity.ok);
    EXPECT_FALSE(validity.requires_curved_geometry_support);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 789);

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
    EXPECT_TRUE(validity.ok);
    EXPECT_FALSE(validity.requires_curved_geometry_support);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 790);

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
        EXPECT_GT(patch.quadrature_measure, 0.0);
    }

    const auto validity = svmp::search::diagnose_cut_topology_validity(
        topology,
        /*high_order_parent_geometry=*/true);
    EXPECT_TRUE(validity.ok);
    EXPECT_FALSE(validity.requires_curved_geometry_support);

    CutIntegrationContext context;
    context.importCutTopology(topology, CutGeometryFrame::Reference, 795);

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
#endif

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TESTS_UNIT_SYSTEMS_TRACEMULTIPLIERTESTMESHES_H
#define SVMP_FE_TESTS_UNIT_SYSTEMS_TRACEMULTIPLIERTESTMESHES_H

/**
 * @file TraceMultiplierTestMeshes.h
 * @brief Structured meshes of the trace-multiplier tests (D54 LM3)
 *
 * The unit square as quadrilaterals (Quad4, or Quad9 for order-2 geometry) or
 * as triangles (each square split along its diagonal), and the unit cube as
 * Hex8 cells, with their boundary faces labelled: x = 0 is 1, x = 1 is 2,
 * y = 0 is 3, y = 1 is 4 (and z = 0 is 5, z = 1 is 6).
 */

#include "Assembly/CutIntegrationContext.h"
#include "Core/Types.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "Spaces/H1Space.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace svmp::FE::test_trace_multiplier {

inline constexpr int kLeft = 1;
inline constexpr int kRight = 2;
inline constexpr int kBottom = 3;
inline constexpr int kTop = 4;
inline constexpr int kBack = 5;
inline constexpr int kFront = 6;
/// The label of every boundary face when a builder is asked for one label.
inline constexpr int kWholeBoundary = 7;

/// Boundary faces of a mesh under construction.
struct FaceArrays {
    std::vector<svmp::CellShape> shapes;
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> conn;
    std::vector<std::array<svmp::index_t, 2>> cells;
    std::vector<int> labels;
    int single_label{-1};

    void add(const svmp::CellShape& shape, std::initializer_list<svmp::index_t> vertices, svmp::index_t cell, int label)
    {
        shapes.push_back(shape);
        conn.insert(conn.end(), vertices);
        offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
        cells.push_back({cell, svmp::INVALID_INDEX});
        labels.push_back(label);
    }

    std::shared_ptr<svmp::Mesh> finish(std::shared_ptr<svmp::MeshBase> base)
    {
        base->set_faces_from_arrays(std::move(shapes), std::move(offsets), std::move(conn), std::move(cells),
                                    svmp::MeshCodim1StorageMode::BoundaryOnly);
        for (std::size_t f = 0; f < labels.size(); ++f) {
            base->set_boundary_label(static_cast<svmp::index_t>(f), single_label >= 0 ? single_label : labels[f]);
        }
        return svmp::create_mesh(std::move(base));
    }
};

/// nx x ny quadrilaterals (Quad4 for order 1, Quad9 for order 2) on (0, lx) x (0, ly);
/// with @p single_label >= 0 every boundary face carries that label.
inline std::shared_ptr<svmp::Mesh> buildQuadGrid(int nx, int ny, Real lx, Real ly, int order, int single_label = -1)
{
    auto base = std::make_shared<svmp::MeshBase>();
    const int px = order * nx;
    const int py = order * ny;
    const auto id = [&](int i, int j) { return static_cast<svmp::index_t>(j * (px + 1) + i); };
    std::vector<svmp::real_t> X;
    for (int j = 0; j <= py; ++j) {
        for (int i = 0; i <= px; ++i) {
            X.push_back(static_cast<svmp::real_t>(lx * static_cast<Real>(i) / static_cast<Real>(px)));
            X.push_back(static_cast<svmp::real_t>(ly * static_cast<Real>(j) / static_cast<Real>(py)));
        }
    }
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> conn;
    std::vector<svmp::CellShape> shapes;
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = order;
    const auto cell_of = [&](int cx, int cy) { return static_cast<svmp::index_t>(cy * nx + cx); };
    for (int cy = 0; cy < ny; ++cy) {
        for (int cx = 0; cx < nx; ++cx) {
            const int i0 = order * cx;
            const int j0 = order * cy;
            if (order == 1) {
                conn.insert(conn.end(), {id(i0, j0), id(i0 + 1, j0), id(i0 + 1, j0 + 1), id(i0, j0 + 1)});
            } else {
                conn.insert(conn.end(), {id(i0, j0), id(i0 + 2, j0), id(i0 + 2, j0 + 2), id(i0, j0 + 2),
                                         id(i0 + 1, j0), id(i0 + 2, j0 + 1), id(i0 + 1, j0 + 2),
                                         id(i0, j0 + 1), id(i0 + 1, j0 + 1)});
            }
            offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
            shapes.push_back(shape);
        }
    }
    base->build_from_arrays(/*spatial_dim=*/2, X, offsets, conn, shapes);

    FaceArrays faces;
    faces.single_label = single_label;
    svmp::CellShape line{};
    line.family = svmp::CellFamily::Line;
    line.num_corners = 2;
    line.order = 1;
    for (int cx = 0; cx < nx; ++cx) {
        faces.add(line, {id(order * cx, 0), id(order * (cx + 1), 0)}, cell_of(cx, 0), kBottom);
    }
    for (int cy = 0; cy < ny; ++cy) {
        faces.add(line, {id(px, order * cy), id(px, order * (cy + 1))}, cell_of(nx - 1, cy), kRight);
    }
    for (int cx = 0; cx < nx; ++cx) {
        faces.add(line, {id(order * (cx + 1), py), id(order * cx, py)}, cell_of(cx, ny - 1), kTop);
    }
    for (int cy = 0; cy < ny; ++cy) {
        faces.add(line, {id(0, order * (cy + 1)), id(0, order * cy)}, cell_of(0, cy), kLeft);
    }
    return faces.finish(std::move(base));
}

/// The n x n grid of the unit square, each square split into two Triangle3
/// cells along its diagonal; fields of any order live on these linear cells.
inline std::shared_ptr<svmp::Mesh> buildTriangleGrid(int n, int single_label = -1)
{
    auto base = std::make_shared<svmp::MeshBase>();
    const auto id = [&](int i, int j) { return static_cast<svmp::index_t>(j * (n + 1) + i); };
    std::vector<svmp::real_t> X;
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            X.push_back(static_cast<svmp::real_t>(static_cast<Real>(i) / static_cast<Real>(n)));
            X.push_back(static_cast<svmp::real_t>(static_cast<Real>(j) / static_cast<Real>(n)));
        }
    }
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> conn;
    std::vector<svmp::CellShape> shapes;
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Triangle;
    shape.num_corners = 3;
    shape.order = 1;
    // Cell 2 (j n + i) is the lower triangle of square (i, j), 2 (j n + i) + 1 the upper.
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            conn.insert(conn.end(), {id(i, j), id(i + 1, j), id(i + 1, j + 1)});
            offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
            conn.insert(conn.end(), {id(i, j), id(i + 1, j + 1), id(i, j + 1)});
            offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
            shapes.push_back(shape);
            shapes.push_back(shape);
        }
    }
    base->build_from_arrays(/*spatial_dim=*/2, X, offsets, conn, shapes);

    FaceArrays faces;
    faces.single_label = single_label;
    svmp::CellShape line{};
    line.family = svmp::CellFamily::Line;
    line.num_corners = 2;
    line.order = 1;
    const auto lower = [&](int i, int j) { return static_cast<svmp::index_t>(2 * (j * n + i)); };
    for (int i = 0; i < n; ++i) {
        faces.add(line, {id(i, 0), id(i + 1, 0)}, lower(i, 0), kBottom);
        faces.add(line, {id(i + 1, n), id(i, n)}, lower(i, n - 1) + 1, kTop);
    }
    for (int j = 0; j < n; ++j) {
        faces.add(line, {id(n, j), id(n, j + 1)}, lower(n - 1, j), kRight);
        faces.add(line, {id(0, j + 1), id(0, j)}, lower(0, j) + 1, kLeft);
    }
    return faces.finish(std::move(base));
}

/// The n x n x n grid of the unit cube as Hex8 cells.
inline std::shared_ptr<svmp::Mesh> buildHexGrid(int n)
{
    auto base = std::make_shared<svmp::MeshBase>();
    const auto id = [&](int i, int j, int k) {
        return static_cast<svmp::index_t>((k * (n + 1) + j) * (n + 1) + i);
    };
    std::vector<svmp::real_t> X;
    for (int k = 0; k <= n; ++k) {
        for (int j = 0; j <= n; ++j) {
            for (int i = 0; i <= n; ++i) {
                X.push_back(static_cast<svmp::real_t>(static_cast<Real>(i) / static_cast<Real>(n)));
                X.push_back(static_cast<svmp::real_t>(static_cast<Real>(j) / static_cast<Real>(n)));
                X.push_back(static_cast<svmp::real_t>(static_cast<Real>(k) / static_cast<Real>(n)));
            }
        }
    }
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> conn;
    std::vector<svmp::CellShape> shapes;
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Hex;
    shape.num_corners = 8;
    shape.order = 1;
    const auto cell_of = [&](int i, int j, int k) { return static_cast<svmp::index_t>((k * n + j) * n + i); };
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                conn.insert(conn.end(), {id(i, j, k), id(i + 1, j, k), id(i + 1, j + 1, k), id(i, j + 1, k),
                                         id(i, j, k + 1), id(i + 1, j, k + 1), id(i + 1, j + 1, k + 1),
                                         id(i, j + 1, k + 1)});
                offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
                shapes.push_back(shape);
            }
        }
    }
    base->build_from_arrays(/*spatial_dim=*/3, X, offsets, conn, shapes);

    FaceArrays faces;
    svmp::CellShape quad{};
    quad.family = svmp::CellFamily::Quad;
    quad.num_corners = 4;
    quad.order = 1;
    // Each face counterclockwise seen from outside the cube.
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            faces.add(quad, {id(0, a, b), id(0, a, b + 1), id(0, a + 1, b + 1), id(0, a + 1, b)}, cell_of(0, a, b),
                      kLeft);
            faces.add(quad, {id(n, a, b), id(n, a + 1, b), id(n, a + 1, b + 1), id(n, a, b + 1)},
                      cell_of(n - 1, a, b), kRight);
            faces.add(quad, {id(a, 0, b), id(a + 1, 0, b), id(a + 1, 0, b + 1), id(a, 0, b + 1)}, cell_of(a, 0, b),
                      kBottom);
            faces.add(quad, {id(a, n, b), id(a, n, b + 1), id(a + 1, n, b + 1), id(a + 1, n, b)},
                      cell_of(a, n - 1, b), kTop);
            faces.add(quad, {id(a, b, 0), id(a, b + 1, 0), id(a + 1, b + 1, 0), id(a + 1, b, 0)}, cell_of(a, b, 0),
                      kBack);
            faces.add(quad, {id(a, b, n), id(a + 1, b, n), id(a + 1, b + 1, n), id(a, b + 1, n)},
                      cell_of(a, b, n - 1), kFront);
        }
    }
    return faces.finish(std::move(base));
}

/**
 * A cut integration context holding the generated interface @p marker of the
 * level set @p phi (sampled at the vertices, linear on each cell) over every
 * cell of a 2D mesh of linear cells, with both cut volumes.
 */
inline std::shared_ptr<assembly::CutIntegrationContext> generatedInterfaceContext(
    const svmp::Mesh& mesh,
    int marker,
    const std::function<Real(Real, Real)>& phi,
    std::uint64_t value_revision = 1u,
    int quadrature_order = 2)
{
    interfaces::CutInterfaceDomainRequest request;
    request.source = interfaces::LevelSetInterfaceSource::fromField(/*field_id=*/0, /*layout_revision=*/1,
                                                                     value_revision);
    request.interface_marker = marker;
    request.quadrature_order = quadrature_order;
    interfaces::LevelSetInterfaceDomain domain(request);
    // The cut is built in the reference cell (the rules' frame); the level set
    // is sampled at the physical vertices.
    const std::vector<std::array<Real, 3>> reference_quad{
        {{-1.0, -1.0, 0.0}}, {{1.0, -1.0, 0.0}}, {{1.0, 1.0, 0.0}}, {{-1.0, 1.0, 0.0}}};
    const std::vector<std::array<Real, 3>> reference_triangle{
        {{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}};
    const auto& base = mesh.local_mesh();
    for (std::size_t c = 0; c < base.n_cells(); ++c) {
        const auto [vertices, count] = base.cell_vertices_span(static_cast<svmp::index_t>(c));
        interfaces::LevelSetCellCutInput input;
        input.parent_cell = static_cast<MeshIndex>(c);
        const bool triangle = count == 3u;
        input.element_type = triangle ? ElementType::Triangle3 : ElementType::Quad4;
        input.node_coordinates = triangle ? reference_triangle : reference_quad;
        for (std::size_t i = 0; i < input.node_coordinates.size(); ++i) {
            const auto x = base.get_vertex_coords(vertices[i]);
            input.level_set_values.push_back(phi(static_cast<Real>(x[0]), static_cast<Real>(x[1])));
        }
        interfaces::appendLinearLevelSetCellCut2D(domain, input);
    }
    auto context = std::make_shared<assembly::CutIntegrationContext>();
    context->addGeneratedInterfaceDomain(domain);
    return context;
}

inline std::shared_ptr<spaces::H1Space> quadSpace(int order)
{
    return std::make_shared<spaces::H1Space>(order == 1 ? ElementType::Quad4 : ElementType::Quad9, order);
}

inline std::shared_ptr<spaces::H1Space> triangleSpace(int order)
{
    return std::make_shared<spaces::H1Space>(ElementType::Triangle3, order);
}

inline std::shared_ptr<spaces::H1Space> hexSpace(int order)
{
    return std::make_shared<spaces::H1Space>(ElementType::Hex8, order);
}

} // namespace svmp::FE::test_trace_multiplier

#endif // SVMP_FE_TESTS_UNIT_SYSTEMS_TRACEMULTIPLIERTESTMESHES_H

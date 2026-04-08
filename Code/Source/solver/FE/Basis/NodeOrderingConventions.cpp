/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "NodeOrderingConventions.h"
#include "Core/FEException.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace svmp {
namespace FE {
namespace basis {

namespace {

using Point = math::Vector<Real, 3>;

ElementType canonical_lagrange_type(ElementType elem_type) {
    switch (elem_type) {
        case ElementType::Line2:
        case ElementType::Line3:
            return ElementType::Line2;
        case ElementType::Triangle3:
        case ElementType::Triangle6:
            return ElementType::Triangle3;
        case ElementType::Quad4:
        case ElementType::Quad9:
            return ElementType::Quad4;
        case ElementType::Tetra4:
        case ElementType::Tetra10:
            return ElementType::Tetra4;
        case ElementType::Hex8:
        case ElementType::Hex27:
            return ElementType::Hex8;
        case ElementType::Wedge6:
        case ElementType::Wedge18:
            return ElementType::Wedge6;
        case ElementType::Pyramid5:
        case ElementType::Pyramid14:
            return ElementType::Pyramid5;
        default:
            return elem_type;
    }
}

int complete_lagrange_alias_order(ElementType elem_type) {
    switch (elem_type) {
        case ElementType::Line2:
        case ElementType::Triangle3:
        case ElementType::Quad4:
        case ElementType::Tetra4:
        case ElementType::Hex8:
        case ElementType::Wedge6:
        case ElementType::Pyramid5:
            return 1;
        case ElementType::Line3:
        case ElementType::Triangle6:
        case ElementType::Quad9:
        case ElementType::Tetra10:
        case ElementType::Hex27:
        case ElementType::Wedge18:
        case ElementType::Pyramid14:
            return 2;
        default:
            return -1;
    }
}

Real line_coord_pm_one(int i, int order) {
    if (order <= 0) {
        return Real(0);
    }
    return Real(-1) + Real(2) * static_cast<Real>(i) / static_cast<Real>(order);
}

Real line_coord_zero_one(int i, int order) {
    if (order <= 0) {
        return Real(0);
    }
    return static_cast<Real>(i) / static_cast<Real>(order);
}

void append_triangle_face_interior(std::vector<Point>& nodes,
                                   const Point& v0,
                                   const Point& v1,
                                   const Point& v2,
                                   int order) {
    for (int c = 1; c <= order - 2; ++c) {
        for (int b = 1; b <= order - c - 1; ++b) {
            const int a = order - b - c;
            const Real la = static_cast<Real>(a) / static_cast<Real>(order);
            const Real lb = static_cast<Real>(b) / static_cast<Real>(order);
            const Real lc = static_cast<Real>(c) / static_cast<Real>(order);
            nodes.push_back(v0 * la + v1 * lb + v2 * lc);
        }
    }
}

std::vector<Point> generate_line_nodes(int order) {
    if (order == 0) {
        return {Point{Real(0), Real(0), Real(0)}};
    }

    std::vector<Point> nodes;
    nodes.reserve(static_cast<std::size_t>(order + 1));
    nodes.push_back(Point{Real(-1), Real(0), Real(0)});
    nodes.push_back(Point{Real(1), Real(0), Real(0)});
    for (int i = 1; i < order; ++i) {
        nodes.push_back(Point{line_coord_pm_one(i, order), Real(0), Real(0)});
    }
    return nodes;
}

std::vector<Point> generate_triangle_nodes(int order) {
    if (order == 0) {
        return {Point{Real(1) / Real(3), Real(1) / Real(3), Real(0)}};
    }

    std::vector<Point> nodes;
    nodes.reserve(static_cast<std::size_t>((order + 1) * (order + 2) / 2));

    nodes.push_back(Point{Real(0), Real(0), Real(0)});
    nodes.push_back(Point{Real(1), Real(0), Real(0)});
    nodes.push_back(Point{Real(0), Real(1), Real(0)});

    for (int m = 1; m < order; ++m) {
        nodes.push_back(Point{line_coord_zero_one(m, order), Real(0), Real(0)});
    }
    for (int m = 1; m < order; ++m) {
        nodes.push_back(Point{line_coord_zero_one(order - m, order),
                              line_coord_zero_one(m, order), Real(0)});
    }
    for (int m = 1; m < order; ++m) {
        nodes.push_back(Point{Real(0), line_coord_zero_one(order - m, order), Real(0)});
    }

    append_triangle_face_interior(
        nodes,
        Point{Real(0), Real(0), Real(0)},
        Point{Real(1), Real(0), Real(0)},
        Point{Real(0), Real(1), Real(0)},
        order);

    return nodes;
}

std::vector<Point> generate_quad_nodes(int order) {
    if (order == 0) {
        return {Point{Real(0), Real(0), Real(0)}};
    }

    std::vector<Point> nodes;
    nodes.reserve(static_cast<std::size_t>((order + 1) * (order + 1)));

    nodes.push_back(Point{Real(-1), Real(-1), Real(0)});
    nodes.push_back(Point{Real(1), Real(-1), Real(0)});
    nodes.push_back(Point{Real(1), Real(1), Real(0)});
    nodes.push_back(Point{Real(-1), Real(1), Real(0)});

    for (int i = 1; i < order; ++i) {
        nodes.push_back(Point{line_coord_pm_one(i, order), Real(-1), Real(0)});
    }
    for (int j = 1; j < order; ++j) {
        nodes.push_back(Point{Real(1), line_coord_pm_one(j, order), Real(0)});
    }
    for (int i = order - 1; i >= 1; --i) {
        nodes.push_back(Point{line_coord_pm_one(i, order), Real(1), Real(0)});
    }
    for (int j = order - 1; j >= 1; --j) {
        nodes.push_back(Point{Real(-1), line_coord_pm_one(j, order), Real(0)});
    }

    for (int j = 1; j < order; ++j) {
        for (int i = 1; i < order; ++i) {
            nodes.push_back(Point{line_coord_pm_one(i, order), line_coord_pm_one(j, order), Real(0)});
        }
    }

    return nodes;
}

std::vector<Point> generate_tetra_nodes(int order) {
    if (order == 0) {
        return {Point{Real(0.25), Real(0.25), Real(0.25)}};
    }

    std::vector<Point> nodes;
    nodes.reserve(static_cast<std::size_t>((order + 1) * (order + 2) * (order + 3) / 6));

    const Point verts[] = {
        Point{Real(0), Real(0), Real(0)},
        Point{Real(1), Real(0), Real(0)},
        Point{Real(0), Real(1), Real(0)},
        Point{Real(0), Real(0), Real(1)},
    };
    for (const auto& v : verts) {
        nodes.push_back(v);
    }

    const int edges[6][2] = {
        {0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}
    };
    for (const auto& edge : edges) {
        for (int m = 1; m < order; ++m) {
            const Real t = static_cast<Real>(m) / static_cast<Real>(order);
            nodes.push_back(verts[edge[0]] * (Real(1) - t) + verts[edge[1]] * t);
        }
    }

    const int faces[4][3] = {
        {0, 1, 2},
        {0, 1, 3},
        {1, 2, 3},
        {0, 2, 3},
    };
    for (const auto& face : faces) {
        append_triangle_face_interior(
            nodes,
            verts[face[0]],
            verts[face[1]],
            verts[face[2]],
            order);
    }

    for (int l = 1; l <= order - 3; ++l) {
        for (int k = 1; k <= order - l - 2; ++k) {
            for (int j = 1; j <= order - l - k - 1; ++j) {
                const int i = order - j - k - l;
                const Real x = static_cast<Real>(j) / static_cast<Real>(order);
                const Real y = static_cast<Real>(k) / static_cast<Real>(order);
                const Real z = static_cast<Real>(l) / static_cast<Real>(order);
                nodes.push_back(Point{x, y, z});
            }
        }
    }

    return nodes;
}

std::vector<Point> generate_hex_nodes(int order) {
    if (order == 0) {
        return {Point{Real(0), Real(0), Real(0)}};
    }

    std::vector<Point> nodes;
    nodes.reserve(static_cast<std::size_t>((order + 1) * (order + 1) * (order + 1)));

    const Point verts[] = {
        Point{Real(-1), Real(-1), Real(-1)},
        Point{Real(1), Real(-1), Real(-1)},
        Point{Real(1), Real(1), Real(-1)},
        Point{Real(-1), Real(1), Real(-1)},
        Point{Real(-1), Real(-1), Real(1)},
        Point{Real(1), Real(-1), Real(1)},
        Point{Real(1), Real(1), Real(1)},
        Point{Real(-1), Real(1), Real(1)},
    };
    for (const auto& v : verts) {
        nodes.push_back(v);
    }

    const int edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };
    for (const auto& edge : edges) {
        for (int m = 1; m < order; ++m) {
            const Real t = static_cast<Real>(m) / static_cast<Real>(order);
            nodes.push_back(verts[edge[0]] * (Real(1) - t) + verts[edge[1]] * t);
        }
    }

    for (int j = 1; j < order; ++j) {
        for (int i = 1; i < order; ++i) {
            nodes.push_back(Point{line_coord_pm_one(i, order), line_coord_pm_one(j, order), Real(-1)});
        }
    }
    for (int j = 1; j < order; ++j) {
        for (int i = 1; i < order; ++i) {
            nodes.push_back(Point{line_coord_pm_one(i, order), line_coord_pm_one(j, order), Real(1)});
        }
    }
    for (int k = 1; k < order; ++k) {
        for (int i = 1; i < order; ++i) {
            nodes.push_back(Point{line_coord_pm_one(i, order), Real(-1), line_coord_pm_one(k, order)});
        }
    }
    for (int k = 1; k < order; ++k) {
        for (int j = 1; j < order; ++j) {
            nodes.push_back(Point{Real(1), line_coord_pm_one(j, order), line_coord_pm_one(k, order)});
        }
    }
    for (int k = 1; k < order; ++k) {
        for (int i = order - 1; i >= 1; --i) {
            nodes.push_back(Point{line_coord_pm_one(i, order), Real(1), line_coord_pm_one(k, order)});
        }
    }
    for (int k = 1; k < order; ++k) {
        for (int j = order - 1; j >= 1; --j) {
            nodes.push_back(Point{Real(-1), line_coord_pm_one(j, order), line_coord_pm_one(k, order)});
        }
    }

    for (int k = 1; k < order; ++k) {
        for (int j = 1; j < order; ++j) {
            for (int i = 1; i < order; ++i) {
                nodes.push_back(Point{line_coord_pm_one(i, order),
                                      line_coord_pm_one(j, order),
                                      line_coord_pm_one(k, order)});
            }
        }
    }

    return nodes;
}

std::vector<Point> generate_wedge_nodes(int order) {
    if (order == 0) {
        return {Point{Real(1) / Real(3), Real(1) / Real(3), Real(0)}};
    }

    std::vector<Point> nodes;
    nodes.reserve(static_cast<std::size_t>((order + 1) * (order + 1) * (order + 2) / 2));

    const Point verts[] = {
        Point{Real(0), Real(0), Real(-1)},
        Point{Real(1), Real(0), Real(-1)},
        Point{Real(0), Real(1), Real(-1)},
        Point{Real(0), Real(0), Real(1)},
        Point{Real(1), Real(0), Real(1)},
        Point{Real(0), Real(1), Real(1)},
    };
    for (const auto& v : verts) {
        nodes.push_back(v);
    }

    const int edges[9][2] = {
        {0, 1}, {1, 2}, {2, 0},
        {3, 4}, {4, 5}, {5, 3},
        {0, 3}, {1, 4}, {2, 5},
    };
    for (const auto& edge : edges) {
        for (int m = 1; m < order; ++m) {
            const Real t = static_cast<Real>(m) / static_cast<Real>(order);
            nodes.push_back(verts[edge[0]] * (Real(1) - t) + verts[edge[1]] * t);
        }
    }

    append_triangle_face_interior(
        nodes, verts[0], verts[1], verts[2], order);
    append_triangle_face_interior(
        nodes, verts[3], verts[4], verts[5], order);

    for (int r = 1; r < order; ++r) {
        const Real z = line_coord_pm_one(r, order);
        for (int m = 1; m < order; ++m) {
            const Real t = static_cast<Real>(m) / static_cast<Real>(order);
            nodes.push_back(Point{t, Real(0), z});
        }
        for (int m = 1; m < order; ++m) {
            const Real t = static_cast<Real>(m) / static_cast<Real>(order);
            nodes.push_back(Point{Real(1) - t, t, z});
        }
        for (int m = 1; m < order; ++m) {
            const Real t = static_cast<Real>(m) / static_cast<Real>(order);
            nodes.push_back(Point{Real(0), Real(1) - t, z});
        }
    }

    for (int r = 1; r < order; ++r) {
        const Real z = line_coord_pm_one(r, order);
        for (int c = 1; c <= order - 2; ++c) {
            for (int b = 1; b <= order - c - 1; ++b) {
                const Real x = static_cast<Real>(b) / static_cast<Real>(order);
                const Real y = static_cast<Real>(c) / static_cast<Real>(order);
                nodes.push_back(Point{x, y, z});
            }
        }
    }

    return nodes;
}

std::vector<Point> generate_pyramid_nodes(int order) {
    if (order == 0) {
        return {Point{Real(0), Real(0), Real(0.25)}};
    }

    std::vector<Point> nodes;
    nodes.reserve(static_cast<std::size_t>((order + 1) * (order + 2) * (2 * order + 3) / 6));

    nodes.push_back(Point{Real(-1), Real(-1), Real(0)});
    nodes.push_back(Point{Real(1), Real(-1), Real(0)});
    nodes.push_back(Point{Real(1), Real(1), Real(0)});
    nodes.push_back(Point{Real(-1), Real(1), Real(0)});
    nodes.push_back(Point{Real(0), Real(0), Real(1)});

    for (int m = 1; m < order; ++m) {
        nodes.push_back(Point{line_coord_pm_one(m, order), Real(-1), Real(0)});
    }
    for (int m = 1; m < order; ++m) {
        nodes.push_back(Point{Real(1), line_coord_pm_one(m, order), Real(0)});
    }
    for (int m = order - 1; m >= 1; --m) {
        nodes.push_back(Point{line_coord_pm_one(m, order), Real(1), Real(0)});
    }
    for (int m = order - 1; m >= 1; --m) {
        nodes.push_back(Point{Real(-1), line_coord_pm_one(m, order), Real(0)});
    }

    for (int level = 1; level < order; ++level) {
        const Real z = static_cast<Real>(level) / static_cast<Real>(order);
        const Real scale = Real(1) - z;
        nodes.push_back(Point{-scale, -scale, z});
        nodes.push_back(Point{scale, -scale, z});
        nodes.push_back(Point{scale, scale, z});
        nodes.push_back(Point{-scale, scale, z});
    }

    for (int j = 1; j < order; ++j) {
        for (int i = 1; i < order; ++i) {
            nodes.push_back(Point{line_coord_pm_one(i, order), line_coord_pm_one(j, order), Real(0)});
        }
    }

    for (int level = 1; level < order - 1; ++level) {
        const int n = order - level;
        const Real z = static_cast<Real>(level) / static_cast<Real>(order);
        const Real scale = Real(1) - z;

        for (int m = 1; m < n; ++m) {
            const Real s = line_coord_pm_one(m, n) * scale;
            nodes.push_back(Point{s, -scale, z});
        }
        for (int m = 1; m < n; ++m) {
            const Real s = line_coord_pm_one(m, n) * scale;
            nodes.push_back(Point{scale, s, z});
        }
        for (int m = n - 1; m >= 1; --m) {
            const Real s = line_coord_pm_one(m, n) * scale;
            nodes.push_back(Point{s, scale, z});
        }
        for (int m = n - 1; m >= 1; --m) {
            const Real s = line_coord_pm_one(m, n) * scale;
            nodes.push_back(Point{-scale, s, z});
        }
    }

    for (int level = 1; level < order - 1; ++level) {
        const int n = order - level;
        const Real z = static_cast<Real>(level) / static_cast<Real>(order);
        const Real scale = Real(1) - z;
        for (int j = 1; j < n; ++j) {
            for (int i = 1; i < n; ++i) {
                nodes.push_back(Point{line_coord_pm_one(i, n) * scale,
                                      line_coord_pm_one(j, n) * scale,
                                      z});
            }
        }
    }

    return nodes;
}

const std::vector<Point>& complete_lagrange_alias_coords(ElementType elem_type) {
    switch (elem_type) {
        case ElementType::Line2: {
            static const auto nodes = generate_line_nodes(1);
            return nodes;
        }
        case ElementType::Line3: {
            static const auto nodes = generate_line_nodes(2);
            return nodes;
        }
        case ElementType::Triangle3: {
            static const auto nodes = generate_triangle_nodes(1);
            return nodes;
        }
        case ElementType::Triangle6: {
            static const auto nodes = generate_triangle_nodes(2);
            return nodes;
        }
        case ElementType::Quad4: {
            static const auto nodes = generate_quad_nodes(1);
            return nodes;
        }
        case ElementType::Quad9: {
            static const auto nodes = generate_quad_nodes(2);
            return nodes;
        }
        case ElementType::Tetra4: {
            static const auto nodes = generate_tetra_nodes(1);
            return nodes;
        }
        case ElementType::Tetra10: {
            static const auto nodes = generate_tetra_nodes(2);
            return nodes;
        }
        case ElementType::Hex8: {
            static const auto nodes = generate_hex_nodes(1);
            return nodes;
        }
        case ElementType::Hex27: {
            static const auto nodes = generate_hex_nodes(2);
            return nodes;
        }
        case ElementType::Wedge6: {
            static const auto nodes = generate_wedge_nodes(1);
            return nodes;
        }
        case ElementType::Wedge18: {
            static const auto nodes = generate_wedge_nodes(2);
            return nodes;
        }
        case ElementType::Pyramid5: {
            static const auto nodes = generate_pyramid_nodes(1);
            return nodes;
        }
        case ElementType::Pyramid14: {
            static const auto nodes = generate_pyramid_nodes(2);
            return nodes;
        }
        default:
            throw FEException("NodeOrdering complete-family alias query only supports generated Lagrange aliases",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

} // namespace

math::Vector<Real, 3> NodeOrdering::get_node_coords(ElementType elem_type, std::size_t local_node) {
    if (complete_lagrange_alias_order(elem_type) >= 0) {
        const auto& nodes = complete_lagrange_alias_coords(elem_type);
        if (local_node < nodes.size()) {
            return nodes[local_node];
        }

        throw FEException("Invalid element type or node index in NodeOrdering::get_node_coords",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    switch (elem_type) {
        // 2D Quad Elements
        case ElementType::Quad8:
            switch (local_node) {
                case 0: return {Real(-1), Real(-1), Real(0)};
                case 1: return {Real(+1), Real(-1), Real(0)};
                case 2: return {Real(+1), Real(+1), Real(0)};
                case 3: return {Real(-1), Real(+1), Real(0)};
                case 4: return {Real(0), Real(-1), Real(0)};
                case 5: return {Real(+1), Real(0), Real(0)};
                case 6: return {Real(0), Real(+1), Real(0)};
                case 7: return {Real(-1), Real(0), Real(0)};
                default: break;
            }
            break;

        case ElementType::Hex20:
            switch (local_node) {
                // Corners
                case 0: return {Real(-1), Real(-1), Real(-1)};
                case 1: return {Real(+1), Real(-1), Real(-1)};
                case 2: return {Real(+1), Real(+1), Real(-1)};
                case 3: return {Real(-1), Real(+1), Real(-1)};
                case 4: return {Real(-1), Real(-1), Real(+1)};
                case 5: return {Real(+1), Real(-1), Real(+1)};
                case 6: return {Real(+1), Real(+1), Real(+1)};
                case 7: return {Real(-1), Real(+1), Real(+1)};
                // Mid-edge on bottom
                case 8:  return {Real(0), Real(-1), Real(-1)};
                case 9:  return {Real(+1), Real(0), Real(-1)};
                case 10: return {Real(0), Real(+1), Real(-1)};
                case 11: return {Real(-1), Real(0), Real(-1)};
                // Mid-edge on top
                case 12: return {Real(0), Real(-1), Real(+1)};
                case 13: return {Real(+1), Real(0), Real(+1)};
                case 14: return {Real(0), Real(+1), Real(+1)};
                case 15: return {Real(-1), Real(0), Real(+1)};
                // Vertical mid-edges
                case 16: return {Real(-1), Real(-1), Real(0)};
                case 17: return {Real(+1), Real(-1), Real(0)};
                case 18: return {Real(+1), Real(+1), Real(0)};
                case 19: return {Real(-1), Real(+1), Real(0)};
                default: break;
            }
            break;

        case ElementType::Wedge15:
            switch (local_node) {
                // Corners
                case 0: return {Real(0), Real(0), Real(-1)};
                case 1: return {Real(1), Real(0), Real(-1)};
                case 2: return {Real(0), Real(1), Real(-1)};
                case 3: return {Real(0), Real(0), Real(+1)};
                case 4: return {Real(1), Real(0), Real(+1)};
                case 5: return {Real(0), Real(1), Real(+1)};
                // Bottom mid-edges
                case 6: return {Real(0.5), Real(0), Real(-1)};
                case 7: return {Real(0.5), Real(0.5), Real(-1)};
                case 8: return {Real(0), Real(0.5), Real(-1)};
                // Top mid-edges
                case 9:  return {Real(0.5), Real(0), Real(+1)};
                case 10: return {Real(0.5), Real(0.5), Real(+1)};
                case 11: return {Real(0), Real(0.5), Real(+1)};
                // Vertical mid-edges
                case 12: return {Real(0), Real(0), Real(0)};
                case 13: return {Real(1), Real(0), Real(0)};
                case 14: return {Real(0), Real(1), Real(0)};
                default: break;
            }
            break;

        case ElementType::Pyramid13:
            switch (local_node) {
                // Base corners
                case 0: return {Real(-1), Real(-1), Real(0)};
                case 1: return {Real(+1), Real(-1), Real(0)};
                case 2: return {Real(+1), Real(+1), Real(0)};
                case 3: return {Real(-1), Real(+1), Real(0)};
                // Apex
                case 4: return {Real(0), Real(0), Real(1)};
                // Base mid-edges
                case 5: return {Real(0), Real(-1), Real(0)};
                case 6: return {Real(+1), Real(0), Real(0)};
                case 7: return {Real(0), Real(+1), Real(0)};
                case 8: return {Real(-1), Real(0), Real(0)};
                // Mid-edges to apex
                case 9:  return {Real(-0.5), Real(-0.5), Real(0.5)};
                case 10: return {Real(+0.5), Real(-0.5), Real(0.5)};
                case 11: return {Real(+0.5), Real(+0.5), Real(0.5)};
                case 12: return {Real(-0.5), Real(+0.5), Real(0.5)};
                default: break;
            }
            break;

        default:
            break;
    }

    throw FEException("Invalid element type or node index in NodeOrdering::get_node_coords",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

std::size_t NodeOrdering::num_nodes(ElementType elem_type) {
    if (complete_lagrange_alias_order(elem_type) >= 0) {
        return complete_lagrange_alias_coords(elem_type).size();
    }

    switch (elem_type) {
        case ElementType::Quad8:      return 8;
        case ElementType::Hex20:      return 20;
        case ElementType::Wedge15:    return 15;
        case ElementType::Pyramid13:  return 13;
        default:
            throw FEException("Unknown element type in NodeOrdering::num_nodes",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

std::vector<math::Vector<Real, 3>>
NodeOrdering::get_lagrange_node_coords(ElementType canonical_type, int order) {
    if (order < 0) {
        throw FEException("NodeOrdering::get_lagrange_node_coords requires non-negative order",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    switch (canonical_lagrange_type(canonical_type)) {
        case ElementType::Point1:
            return {Point{Real(0), Real(0), Real(0)}};
        case ElementType::Line2:
            return generate_line_nodes(order);
        case ElementType::Triangle3:
            return generate_triangle_nodes(order);
        case ElementType::Quad4:
            return generate_quad_nodes(order);
        case ElementType::Tetra4:
            return generate_tetra_nodes(order);
        case ElementType::Hex8:
            return generate_hex_nodes(order);
        case ElementType::Wedge6:
            return generate_wedge_nodes(order);
        case ElementType::Pyramid5:
            return generate_pyramid_nodes(order);
        case ElementType::Quad8:
        case ElementType::Hex20:
        case ElementType::Wedge15:
        case ElementType::Pyramid13:
            throw FEException("NodeOrdering::get_lagrange_node_coords does not support serendipity topologies",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        default:
            throw FEException("NodeOrdering::get_lagrange_node_coords: unsupported topology",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

bool NodeOrdering::is_simplex(ElementType elem_type) {
    switch (elem_type) {
        case ElementType::Triangle3:
        case ElementType::Triangle6:
        case ElementType::Tetra4:
        case ElementType::Tetra10:
            return true;
        default:
            return false;
    }
}

bool NodeOrdering::is_tensor_product(ElementType elem_type) {
    switch (elem_type) {
        case ElementType::Line2:
        case ElementType::Line3:
        case ElementType::Quad4:
        case ElementType::Quad8:
        case ElementType::Quad9:
        case ElementType::Hex8:
        case ElementType::Hex20:
        case ElementType::Hex27:
            return true;
        default:
            return false;
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp

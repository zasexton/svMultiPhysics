/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Elements/ElementTransform.h"
#include "Elements/ReferenceElement.h"
#include "Basis/NodeOrderingConventions.h"

#include <cmath>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace elements {

using svmp::FE::geometry::PushForward;

void ElementTransform::gradients_to_physical(const geometry::GeometryMapping& mapping,
                                             const math::Vector<Real, 3>& xi,
                                             const std::vector<basis::Gradient>& grads_ref,
                                             std::vector<math::Vector<Real, 3>>& grads_phys) {
    const std::size_t n = grads_ref.size();
    grads_phys.resize(n);

    for (std::size_t i = 0; i < n; ++i) {
        math::Vector<Real, 3> g_ref{};
        const int dim = mapping.dimension();
        for (int d = 0; d < dim; ++d) {
            g_ref[static_cast<std::size_t>(d)] = grads_ref[i][static_cast<std::size_t>(d)];
        }
        grads_phys[i] = PushForward::gradient(mapping, g_ref, xi);
    }
}

void ElementTransform::hdiv_vectors_to_physical(const geometry::GeometryMapping& mapping,
                                                const math::Vector<Real, 3>& xi,
                                                const std::vector<math::Vector<Real, 3>>& v_ref,
                                                std::vector<math::Vector<Real, 3>>& v_phys) {
    const std::size_t n = v_ref.size();
    v_phys.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        v_phys[i] = PushForward::hdiv_vector(mapping, v_ref[i], xi);
    }
}

void ElementTransform::hcurl_vectors_to_physical(const geometry::GeometryMapping& mapping,
                                                 const math::Vector<Real, 3>& xi,
                                                 const std::vector<math::Vector<Real, 3>>& v_ref,
                                                 std::vector<math::Vector<Real, 3>>& v_phys) {
    const std::size_t n = v_ref.size();
    v_phys.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        v_phys[i] = PushForward::hcurl_vector(mapping, v_ref[i], xi);
    }
}

// =============================================================================
// Facet Frame and Trace Evaluation Implementation
// =============================================================================

namespace {

// Helper: normalize a vector and return its length
Real normalize(math::Vector<Real, 3>& v) {
    Real len = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (len > Real(1e-14)) {
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }
    return len;
}

// Helper: cross product
math::Vector<Real, 3> cross(const math::Vector<Real, 3>& a,
                            const math::Vector<Real, 3>& b) {
    return math::Vector<Real, 3>{
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    };
}

// Helper: dot product
Real dot(const math::Vector<Real, 3>& a, const math::Vector<Real, 3>& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

} // anonymous namespace

FacetFrame ElementTransform::compute_facet_frame(const geometry::GeometryMapping& mapping,
                                                  const math::Vector<Real, 3>& xi,
                                                  int facet_id,
                                                  ElementType element_type) {
    FacetFrame frame;
    const int dim = element_dimension(element_type);

    // Get the reference facet normal (outward direction)
    math::Vector<Real, 3> ref_normal = reference_facet_normal(element_type, facet_id);

    // Get Jacobian at the evaluation point
    auto J = mapping.jacobian(xi);

    if (dim == 2) {
        // For 2D elements, facet is an edge
        // Tangent is perpendicular to normal in reference space
        frame.tangent1 = math::Vector<Real, 3>{-ref_normal[1], ref_normal[0], Real(0)};

        // Map tangent to physical space: t_phys = J * t_ref
        math::Vector<Real, 3> t_phys{};
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 2; ++j) {
                t_phys[i] += J(i, j) * frame.tangent1[j];
            }
        }
        frame.jacobian_det = normalize(t_phys);
        frame.tangent1 = t_phys;

        // Normal is 90-degree rotation of tangent in 2D
        frame.normal = math::Vector<Real, 3>{-frame.tangent1[1], frame.tangent1[0], Real(0)};

        // Ensure outward orientation by checking against reference normal mapped through J
        math::Vector<Real, 3> mapped_ref_normal{};
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 2; ++j) {
                mapped_ref_normal[i] += J(i, j) * ref_normal[j];
            }
        }
        if (dot(frame.normal, mapped_ref_normal) < Real(0)) {
            frame.normal[0] = -frame.normal[0];
            frame.normal[1] = -frame.normal[1];
        }
    } else if (dim == 3) {
        // For 3D elements, facet is a face
        // Get two tangent directions on the face (from reference face parameterization)
        auto [vertices, coords] = facet_vertices(element_type, facet_id);

        if (coords.size() < 3) {
            // Degenerate case - shouldn't happen for valid 3D elements
            frame.normal = ref_normal;
            frame.tangent1 = math::Vector<Real, 3>{Real(1), Real(0), Real(0)};
            frame.tangent2 = math::Vector<Real, 3>{Real(0), Real(1), Real(0)};
            frame.jacobian_det = Real(1);
            return frame;
        }

        // Reference tangents from vertex coordinates
        math::Vector<Real, 3> t1_ref{
            coords[1][0] - coords[0][0],
            coords[1][1] - coords[0][1],
            coords[1][2] - coords[0][2]
        };
        math::Vector<Real, 3> t2_ref{
            coords[2][0] - coords[0][0],
            coords[2][1] - coords[0][1],
            coords[2][2] - coords[0][2]
        };

        // Map tangents to physical space
        math::Vector<Real, 3> t1_phys{}, t2_phys{};
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                t1_phys[i] += J(i, j) * t1_ref[j];
                t2_phys[i] += J(i, j) * t2_ref[j];
            }
        }

        // Physical normal is cross product of tangents
        frame.normal = cross(t1_phys, t2_phys);
        frame.jacobian_det = normalize(frame.normal);

        // Ensure outward orientation
        math::Vector<Real, 3> mapped_ref_normal{};
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                mapped_ref_normal[i] += J(i, j) * ref_normal[j];
            }
        }
        if (dot(frame.normal, mapped_ref_normal) < Real(0)) {
            frame.normal[0] = -frame.normal[0];
            frame.normal[1] = -frame.normal[1];
            frame.normal[2] = -frame.normal[2];
        }

        // Normalize tangents
        normalize(t1_phys);
        normalize(t2_phys);
        frame.tangent1 = t1_phys;
        frame.tangent2 = t2_phys;
    }

    return frame;
}

std::vector<Real> ElementTransform::hdiv_normal_trace(
    const std::vector<math::Vector<Real, 3>>& v_phys,
    const math::Vector<Real, 3>& normal) {

    std::vector<Real> traces(v_phys.size());
    for (std::size_t i = 0; i < v_phys.size(); ++i) {
        traces[i] = dot(v_phys[i], normal);
    }
    return traces;
}

std::vector<math::Vector<Real, 3>> ElementTransform::hcurl_tangential_trace_3d(
    const std::vector<math::Vector<Real, 3>>& v_phys,
    const FacetFrame& frame) {

    std::vector<math::Vector<Real, 3>> traces(v_phys.size());
    for (std::size_t i = 0; i < v_phys.size(); ++i) {
        // Tangential trace is n x v (component in tangent plane)
        traces[i] = cross(frame.normal, v_phys[i]);
    }
    return traces;
}

std::vector<Real> ElementTransform::hcurl_tangential_trace_2d(
    const std::vector<math::Vector<Real, 3>>& v_phys,
    const FacetFrame& frame) {

    std::vector<Real> traces(v_phys.size());
    for (std::size_t i = 0; i < v_phys.size(); ++i) {
        // In 2D, tangential trace is t . v
        traces[i] = dot(v_phys[i], frame.tangent1);
    }
    return traces;
}

std::pair<std::vector<LocalIndex>, std::vector<math::Vector<Real, 3>>>
ElementTransform::facet_vertices(ElementType element_type, int facet_id) {
    ReferenceElement ref = ReferenceElement::create(element_type);
    const int dim = element_dimension(element_type);

    std::vector<LocalIndex> vertex_indices;
    std::vector<math::Vector<Real, 3>> coords;

    if (dim == 2) {
        // Facets are edges
        if (static_cast<std::size_t>(facet_id) >= ref.num_faces()) {
            return {vertex_indices, coords};
        }
        const auto& edge_nodes = ref.face_nodes(static_cast<std::size_t>(facet_id));
        vertex_indices.assign(edge_nodes.begin(), edge_nodes.end());
    } else if (dim == 3) {
        // Facets are faces
        if (static_cast<std::size_t>(facet_id) >= ref.num_faces()) {
            return {vertex_indices, coords};
        }
        const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(facet_id));
        vertex_indices.assign(face_nodes.begin(), face_nodes.end());
    }

    // Get reference coordinates for each vertex
    const std::size_t num_nodes = basis::NodeOrdering::num_nodes(element_type);
    for (LocalIndex vi : vertex_indices) {
        if (vi < num_nodes) {
            coords.push_back(basis::NodeOrdering::get_node_coords(element_type, vi));
        }
    }

    return {vertex_indices, coords};
}

math::Vector<Real, 3> ElementTransform::facet_to_reference(
    ElementType element_type,
    int facet_id,
    const math::Vector<Real, 3>& facet_coords) {

    auto [vertices, coords] = facet_vertices(element_type, facet_id);
    const int dim = element_dimension(element_type);

    if (coords.empty()) {
        return math::Vector<Real, 3>{};
    }

    math::Vector<Real, 3> result{};

    if (dim == 2) {
        // Edge: linear interpolation using facet_coords[0] as parameter t in [0,1]
        if (coords.size() >= 2) {
            Real t = facet_coords[0];
            for (std::size_t i = 0; i < 3; ++i) {
                result[i] = (Real(1) - t) * coords[0][i] + t * coords[1][i];
            }
        }
    } else if (dim == 3) {
        // Face: depends on face shape
        if (coords.size() == 3) {
            // Triangle face: barycentric coordinates (facet_coords[0], facet_coords[1])
            // where third coordinate is 1 - facet_coords[0] - facet_coords[1]
            Real L0 = facet_coords[0];
            Real L1 = facet_coords[1];
            Real L2 = Real(1) - L0 - L1;
            for (std::size_t i = 0; i < 3; ++i) {
                result[i] = L2 * coords[0][i] + L0 * coords[1][i] + L1 * coords[2][i];
            }
        } else if (coords.size() >= 4) {
            // Quad face: tensor-product coordinates (s, t) in [0,1]^2
            Real s = facet_coords[0];
            Real t = facet_coords[1];
            for (std::size_t i = 0; i < 3; ++i) {
                result[i] = (Real(1) - s) * (Real(1) - t) * coords[0][i]
                          + s * (Real(1) - t) * coords[1][i]
                          + s * t * coords[2][i]
                          + (Real(1) - s) * t * coords[3][i];
            }
        }
    }

    return result;
}

math::Vector<Real, 3> ElementTransform::reference_facet_normal(
    ElementType element_type,
    int facet_id) {

    const int dim = element_dimension(element_type);
    math::Vector<Real, 3> normal{};

    // Define outward normals for standard reference elements
    // These are the canonical outward normals for each facet

    if (dim == 2) {
        // 2D elements: Triangle and Quad
        switch (element_type) {
            case ElementType::Triangle3:
            case ElementType::Triangle6:
                // Reference triangle: vertices at (0,0), (1,0), (0,1)
                // Edge 0: (0,0)-(1,0), normal = (0,-1)
                // Edge 1: (1,0)-(0,1), normal = (1,1)/sqrt(2)
                // Edge 2: (0,1)-(0,0), normal = (-1,0)
                switch (facet_id) {
                    case 0: normal = {Real(0), Real(-1), Real(0)}; break;
                    case 1: normal = {Real(1)/std::sqrt(Real(2)), Real(1)/std::sqrt(Real(2)), Real(0)}; break;
                    case 2: normal = {Real(-1), Real(0), Real(0)}; break;
                    default: break;
                }
                break;

            case ElementType::Quad4:
            case ElementType::Quad8:
            case ElementType::Quad9:
                // Reference quad: vertices at (-1,-1), (1,-1), (1,1), (-1,1)
                // Edge 0: (-1,-1)-(1,-1), normal = (0,-1)
                // Edge 1: (1,-1)-(1,1), normal = (1,0)
                // Edge 2: (1,1)-(-1,1), normal = (0,1)
                // Edge 3: (-1,1)-(-1,-1), normal = (-1,0)
                switch (facet_id) {
                    case 0: normal = {Real(0), Real(-1), Real(0)}; break;
                    case 1: normal = {Real(1), Real(0), Real(0)}; break;
                    case 2: normal = {Real(0), Real(1), Real(0)}; break;
                    case 3: normal = {Real(-1), Real(0), Real(0)}; break;
                    default: break;
                }
                break;

            default:
                break;
        }
    } else if (dim == 3) {
        // 3D elements
        switch (element_type) {
            case ElementType::Tetra4:
            case ElementType::Tetra10:
                // Reference tetrahedron: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
                // Face 0: (0,0,0)-(1,0,0)-(0,1,0), normal = (0,0,-1)
                // Face 1: (0,0,0)-(1,0,0)-(0,0,1), normal = (0,-1,0)
                // Face 2: (1,0,0)-(0,1,0)-(0,0,1), normal = outward from origin
                // Face 3: (0,0,0)-(0,1,0)-(0,0,1), normal = (-1,0,0)
                switch (facet_id) {
                    case 0: normal = {Real(0), Real(0), Real(-1)}; break;
                    case 1: normal = {Real(0), Real(-1), Real(0)}; break;
                    case 2: {
                        // Compute outward normal for oblique face
                        Real s = Real(1) / std::sqrt(Real(3));
                        normal = {s, s, s};
                        break;
                    }
                    case 3: normal = {Real(-1), Real(0), Real(0)}; break;
                    default: break;
                }
                break;

            case ElementType::Hex8:
            case ElementType::Hex20:
            case ElementType::Hex27:
                // Reference hex: [-1,1]^3
                // Face 0: z = -1, normal = (0,0,-1)
                // Face 1: z = +1, normal = (0,0,+1)
                // Face 2: y = -1, normal = (0,-1,0)
                // Face 3: x = +1, normal = (+1,0,0)
                // Face 4: y = +1, normal = (0,+1,0)
                // Face 5: x = -1, normal = (-1,0,0)
                switch (facet_id) {
                    case 0: normal = {Real(0), Real(0), Real(-1)}; break;
                    case 1: normal = {Real(0), Real(0), Real(1)}; break;
                    case 2: normal = {Real(0), Real(-1), Real(0)}; break;
                    case 3: normal = {Real(1), Real(0), Real(0)}; break;
                    case 4: normal = {Real(0), Real(1), Real(0)}; break;
                    case 5: normal = {Real(-1), Real(0), Real(0)}; break;
                    default: break;
                }
                break;

            case ElementType::Wedge6:
            case ElementType::Wedge15:
            case ElementType::Wedge18:
                // Wedge: triangular bases at z=0 and z=1, three quad sides
                // Face 0: bottom triangle z=0, normal = (0,0,-1)
                // Face 1: top triangle z=1, normal = (0,0,+1)
                // Face 2-4: quad sides
                switch (facet_id) {
                    case 0: normal = {Real(0), Real(0), Real(-1)}; break;
                    case 1: normal = {Real(0), Real(0), Real(1)}; break;
                    case 2: normal = {Real(0), Real(-1), Real(0)}; break;
                    case 3: {
                        Real s = Real(1) / std::sqrt(Real(2));
                        normal = {s, s, Real(0)};
                        break;
                    }
                    case 4: normal = {Real(-1), Real(0), Real(0)}; break;
                    default: break;
                }
                break;

            case ElementType::Pyramid5:
            case ElementType::Pyramid13:
            case ElementType::Pyramid14:
                // Pyramid: quad base at z=0, apex at (0,0,1)
                // Face 0: quad base, normal = (0,0,-1)
                // Face 1-4: triangular sides
                switch (facet_id) {
                    case 0: normal = {Real(0), Real(0), Real(-1)}; break;
                    case 1: normal = {Real(0), Real(-1), Real(1)}; normalize(normal); break;
                    case 2: normal = {Real(1), Real(0), Real(1)}; normalize(normal); break;
                    case 3: normal = {Real(0), Real(1), Real(1)}; normalize(normal); break;
                    case 4: normal = {Real(-1), Real(0), Real(1)}; normalize(normal); break;
                    default: break;
                }
                break;

            default:
                break;
        }
    }

    return normal;
}

} // namespace elements
} // namespace FE
} // namespace svmp


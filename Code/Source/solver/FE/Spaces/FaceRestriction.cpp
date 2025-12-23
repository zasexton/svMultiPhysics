/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/FaceRestriction.h"
#include "Core/FEException.h"
#include "Basis/LagrangeBasis.h"
#include "Basis/NodeOrderingConventions.h"
#include "Elements/ReferenceElement.h"
#include <algorithm>

namespace svmp {
namespace FE {
namespace spaces {

namespace {

using Vec3 = math::Vector<Real, 3>;

constexpr Real kGeomTol = Real(1e-12);

bool is_serendipity_element(ElementType type) {
    switch (type) {
        case ElementType::Quad8:
        case ElementType::Hex20:
        case ElementType::Wedge15:
        case ElementType::Pyramid13:
            return true;
        default:
            return false;
    }
}

std::vector<Vec3> build_dof_nodes(ElementType elem_type, int order) {
    if (is_serendipity_element(elem_type)) {
        FE_CHECK_ARG(order == 2,
                     "FaceRestriction currently supports serendipity elements only at quadratic order 2");
        const std::size_t nn = basis::NodeOrdering::num_nodes(elem_type);
        std::vector<Vec3> nodes;
        nodes.reserve(nn);
        for (std::size_t i = 0; i < nn; ++i) {
            nodes.push_back(basis::NodeOrdering::get_node_coords(elem_type, i));
        }
        return nodes;
    }

    basis::LagrangeBasis basis(elem_type, order);
    const auto& bnodes = basis.nodes();
    return std::vector<Vec3>(bnodes.begin(), bnodes.end());
}

bool point_on_face(const Vec3& x,
                   const std::vector<Vec3>& face_vertices) {
    if (face_vertices.empty()) {
        return false;
    }
    if (face_vertices.size() == 1) {
        return x.approx_equal(face_vertices[0], kGeomTol);
    }

    const Vec3 v0 = face_vertices[0];
    const Vec3 v1 = face_vertices[1];
    const Vec3 e01 = v1 - v0;
    const Vec3 dx = x - v0;

    if (face_vertices.size() == 2) {
        // 2D edge (line segment) embedded in 3D with z=0 for 2D elements.
        const Vec3 n{ -e01[1], e01[0], Real(0) };
        return std::abs(n.dot(dx)) <= kGeomTol;
    }

    // Planar face (triangle/quad) embedded in 3D.
    const Vec3 v2 = face_vertices[2];
    const Vec3 e02 = v2 - v0;
    const Vec3 n = e01.cross(e02);
    return std::abs(n.dot(dx)) <= kGeomTol;
}

bool point_on_edge(const Vec3& x,
                   const Vec3& v0,
                   const Vec3& v1) {
    const Vec3 d = v1 - v0;
    const Real d2 = d.dot(d);
    if (d2 <= Real(0)) {
        return false;
    }

    const Vec3 w = x - v0;
    const Vec3 c = d.cross(w);
    if (c.norm() > kGeomTol) {
        return false;
    }

    const Real t = w.dot(d) / d2;
    return t >= -kGeomTol && t <= Real(1) + kGeomTol;
}

ElementType face_type_from_vertex_count(std::size_t nverts) {
    switch (nverts) {
        case 1: return ElementType::Point1;
        case 2: return ElementType::Line2;
        case 3: return ElementType::Triangle3;
        case 4: return ElementType::Quad4;
        default: return ElementType::Unknown;
    }
}

} // namespace

// Static cache
std::map<FaceRestrictionFactory::CacheKey, std::shared_ptr<const FaceRestriction>>
    FaceRestrictionFactory::cache_;

// =============================================================================
// FaceRestriction Implementation
// =============================================================================

FaceRestriction::FaceRestriction(const FunctionSpace& space)
    : FaceRestriction(space.element_type(), space.polynomial_order(), space.continuity()) {
}

FaceRestriction::FaceRestriction(ElementType elem_type, int polynomial_order, Continuity continuity)
    : elem_type_(elem_type)
    , order_(polynomial_order)
    , continuity_(continuity)
    , num_dofs_(0) {

    FE_CHECK_ARG(is_supported(elem_type),
        "Unsupported element type for FaceRestriction");
    FE_CHECK_ARG(polynomial_order >= 0,
        "Polynomial order must be non-negative");

    initialize_topology();
    compute_dof_maps();
}

bool FaceRestriction::is_supported(ElementType elem_type) {
    switch (elem_type) {
        case ElementType::Line2:
        case ElementType::Line3:
        case ElementType::Triangle3:
        case ElementType::Triangle6:
        case ElementType::Quad4:
        case ElementType::Quad8:
        case ElementType::Quad9:
        case ElementType::Tetra4:
        case ElementType::Tetra10:
        case ElementType::Hex8:
        case ElementType::Hex20:
        case ElementType::Hex27:
        case ElementType::Wedge6:
        case ElementType::Wedge15:
        case ElementType::Wedge18:
        case ElementType::Pyramid5:
        case ElementType::Pyramid13:
        case ElementType::Pyramid14:
            return true;
        default:
            return false;
    }
}

void FaceRestriction::initialize_topology() {
    const elements::ReferenceElement ref = elements::ReferenceElement::create(elem_type_);

    topology_.dimension = ref.dimension();
    topology_.num_edges = static_cast<int>(ref.num_edges());
    topology_.num_faces = static_cast<int>(ref.num_faces());

    topology_.edge_vertices.clear();
    topology_.face_vertices.clear();
    topology_.face_types.clear();

    int max_vertex = -1;

    for (std::size_t e = 0; e < ref.num_edges(); ++e) {
        const auto& nodes = ref.edge_nodes(e);
        FE_CHECK_ARG(nodes.size() == 2, "ReferenceElement edge must have 2 nodes");
        topology_.edge_vertices.emplace_back(static_cast<int>(nodes[0]), static_cast<int>(nodes[1]));
        max_vertex = std::max(max_vertex, static_cast<int>(std::max(nodes[0], nodes[1])));
    }

    for (std::size_t f = 0; f < ref.num_faces(); ++f) {
        const auto& nodes = ref.face_nodes(f);
        std::vector<int> verts;
        verts.reserve(nodes.size());
        for (auto v : nodes) {
            verts.push_back(static_cast<int>(v));
            max_vertex = std::max(max_vertex, static_cast<int>(v));
        }
        topology_.face_vertices.push_back(std::move(verts));
        topology_.face_types.push_back(face_type_from_vertex_count(nodes.size()));
    }

    topology_.num_vertices = max_vertex >= 0 ? (max_vertex + 1) : 0;
}

void FaceRestriction::compute_dof_maps() {
    const auto nodes = build_dof_nodes(elem_type_, order_);
    dof_nodes_ = nodes;
    num_dofs_ = nodes.size();

    face_dof_maps_.assign(static_cast<std::size_t>(topology_.num_faces), {});
    edge_dof_maps_.assign(static_cast<std::size_t>(topology_.num_edges), {});
    vertex_dof_maps_.assign(static_cast<std::size_t>(topology_.num_vertices), {});
    interior_dof_map_.clear();

    // Vertex maps: locate a node coincident with each reference vertex.
    for (int v = 0; v < topology_.num_vertices; ++v) {
        const Vec3 vx = basis::NodeOrdering::get_node_coords(elem_type_, static_cast<std::size_t>(v));
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].approx_equal(vx, kGeomTol)) {
                vertex_dof_maps_[static_cast<std::size_t>(v)].push_back(static_cast<int>(i));
                break;
            }
        }
    }

    // Edge maps: collect all DOFs whose reference node lies on the edge segment.
    for (int e = 0; e < topology_.num_edges; ++e) {
        const auto& edge = topology_.edge_vertices[static_cast<std::size_t>(e)];
        const Vec3 v0 = basis::NodeOrdering::get_node_coords(elem_type_, static_cast<std::size_t>(edge.first));
        const Vec3 v1 = basis::NodeOrdering::get_node_coords(elem_type_, static_cast<std::size_t>(edge.second));

        auto& edofs = edge_dof_maps_[static_cast<std::size_t>(e)];
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (point_on_edge(nodes[i], v0, v1)) {
                edofs.push_back(static_cast<int>(i));
            }
        }
        std::sort(edofs.begin(), edofs.end());
        edofs.erase(std::unique(edofs.begin(), edofs.end()), edofs.end());
    }

    // Face maps: collect all DOFs whose reference node lies on the face plane.
    for (int f = 0; f < topology_.num_faces; ++f) {
        const auto& fverts = topology_.face_vertices[static_cast<std::size_t>(f)];
        std::vector<Vec3> fcoords;
        fcoords.reserve(fverts.size());
        for (int v : fverts) {
            fcoords.push_back(basis::NodeOrdering::get_node_coords(elem_type_, static_cast<std::size_t>(v)));
        }

        auto& fdofs = face_dof_maps_[static_cast<std::size_t>(f)];
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (point_on_face(nodes[i], fcoords)) {
                fdofs.push_back(static_cast<int>(i));
            }
        }
        std::sort(fdofs.begin(), fdofs.end());
        fdofs.erase(std::unique(fdofs.begin(), fdofs.end()), fdofs.end());
    }

    // Interior dofs: those not present on any face.
    std::vector<bool> on_boundary(num_dofs_, false);
    for (const auto& fdofs : face_dof_maps_) {
        for (int i : fdofs) {
            if (i >= 0) {
                on_boundary[static_cast<std::size_t>(i)] = true;
            }
        }
    }
    for (std::size_t i = 0; i < num_dofs_; ++i) {
        if (!on_boundary[i]) {
            interior_dof_map_.push_back(static_cast<int>(i));
        }
    }
}

std::vector<int> FaceRestriction::face_dofs(int face_id) const {
    FE_CHECK_ARG(face_id >= 0 && face_id < static_cast<int>(face_dof_maps_.size()),
        "Invalid face ID");
    return face_dof_maps_[static_cast<std::size_t>(face_id)];
}

std::vector<int> FaceRestriction::edge_dofs(int edge_id) const {
    FE_CHECK_ARG(edge_id >= 0 && edge_id < static_cast<int>(edge_dof_maps_.size()),
        "Invalid edge ID");
    return edge_dof_maps_[static_cast<std::size_t>(edge_id)];
}

std::vector<int> FaceRestriction::vertex_dofs(int vertex_id) const {
    FE_CHECK_ARG(vertex_id >= 0 && vertex_id < static_cast<int>(vertex_dof_maps_.size()),
        "Invalid vertex ID");
    return vertex_dof_maps_[static_cast<std::size_t>(vertex_id)];
}

std::vector<int> FaceRestriction::interior_dofs() const {
    return interior_dof_map_;
}

std::vector<Real> FaceRestriction::restrict_to_face(
    const std::vector<Real>& element_values,
    int face_id) const {

    FE_CHECK_ARG(element_values.size() >= num_dofs_,
        "Element values array too small");

    const auto& dofs = face_dofs(face_id);
    std::vector<Real> result(dofs.size());

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        result[i] = element_values[static_cast<std::size_t>(dofs[i])];
    }

    return result;
}

std::vector<Real> FaceRestriction::restrict_to_edge(
    const std::vector<Real>& element_values,
    int edge_id) const {

    FE_CHECK_ARG(element_values.size() >= num_dofs_,
        "Element values array too small");

    const auto& dofs = edge_dofs(edge_id);
    std::vector<Real> result(dofs.size());

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        result[i] = element_values[static_cast<std::size_t>(dofs[i])];
    }

    return result;
}

void FaceRestriction::scatter_from_face(
    const std::vector<Real>& face_values,
    int face_id,
    std::vector<Real>& element_values) const {

    const auto& dofs = face_dofs(face_id);
    FE_CHECK_ARG(face_values.size() == dofs.size(),
        "Face values size mismatch");
    FE_CHECK_ARG(element_values.size() >= num_dofs_,
        "Element values array too small");

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        element_values[static_cast<std::size_t>(dofs[i])] += face_values[i];
    }
}

void FaceRestriction::scatter_from_edge(
    const std::vector<Real>& edge_values,
    int edge_id,
    std::vector<Real>& element_values) const {

    const auto& dofs = edge_dofs(edge_id);
    FE_CHECK_ARG(edge_values.size() == dofs.size(),
        "Edge values size mismatch");
    FE_CHECK_ARG(element_values.size() >= num_dofs_,
        "Element values array too small");

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        element_values[static_cast<std::size_t>(dofs[i])] += edge_values[i];
    }
}

std::size_t FaceRestriction::num_face_dofs(int face_id) const {
    FE_CHECK_ARG(face_id >= 0 && face_id < static_cast<int>(face_dof_maps_.size()),
        "Invalid face ID");
    return face_dof_maps_[static_cast<std::size_t>(face_id)].size();
}

std::size_t FaceRestriction::num_edge_dofs(int edge_id) const {
    FE_CHECK_ARG(edge_id >= 0 && edge_id < static_cast<int>(edge_dof_maps_.size()),
        "Invalid edge ID");
    return edge_dof_maps_[static_cast<std::size_t>(edge_id)].size();
}

// =============================================================================
// FaceRestrictionFactory Implementation
// =============================================================================

bool FaceRestrictionFactory::CacheKey::operator<(const CacheKey& other) const {
    if (elem_type != other.elem_type) return elem_type < other.elem_type;
    if (order != other.order) return order < other.order;
    return continuity < other.continuity;
}

std::shared_ptr<const FaceRestriction> FaceRestrictionFactory::get(
    ElementType elem_type,
    int order,
    Continuity continuity) {

    CacheKey key{elem_type, order, continuity};

    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second;
    }

    auto restriction = std::make_shared<FaceRestriction>(elem_type, order, continuity);
    cache_[key] = restriction;
    return restriction;
}

std::shared_ptr<const FaceRestriction> FaceRestrictionFactory::get(const FunctionSpace& space) {
    return get(space.element_type(), space.polynomial_order(), space.continuity());
}

void FaceRestrictionFactory::clear_cache() {
    cache_.clear();
}

} // namespace spaces
} // namespace FE
} // namespace svmp

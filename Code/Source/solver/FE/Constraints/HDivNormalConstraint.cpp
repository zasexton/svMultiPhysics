/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/HDivNormalConstraint.h"

#include "Basis/VectorBasis.h"
#include "Elements/ReferenceElement.h"
#include "Forms/PointEvaluator.h"
#include "Geometry/MappingFactory.h"
#include "Quadrature/QuadratureFactory.h"
#include "Spaces/OrientationManager.h"
#include "Spaces/TraceSpace.h"
#include "Systems/FESystem.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <utility>

namespace svmp {
namespace FE {
namespace constraints {

using systems::FESystem;
using systems::InvalidStateException;

HDivNormalConstraint::HDivNormalConstraint(FieldId field, int boundary_marker)
    : HDivNormalConstraint(field, boundary_marker, forms::FormExpr::constant(0.0))
{
}

HDivNormalConstraint::HDivNormalConstraint(FieldId field,
                                           int boundary_marker,
                                           forms::FormExpr value)
    : field_(field)
    , boundary_marker_(boundary_marker)
    , value_(std::move(value))
{
    if (field_ == INVALID_FIELD_ID) {
        throw std::invalid_argument("HDivNormalConstraint: invalid FieldId");
    }
    if (boundary_marker_ < 0) {
        throw std::invalid_argument("HDivNormalConstraint: boundary_marker must be >= 0");
    }
    if (!value_.isValid()) {
        throw std::invalid_argument("HDivNormalConstraint: invalid value expression");
    }
    is_time_dependent_ = forms::isTimeDependent(value_);
}

namespace {

/// Inverts the n x n row-major matrix @p a by Gauss-Jordan elimination with
/// partial pivoting.
[[nodiscard]] std::vector<Real> invertSmallMatrix(std::vector<Real> a, std::size_t n)
{
    std::vector<Real> inv(n * n, Real(0));
    for (std::size_t i = 0; i < n; ++i) {
        inv[i * n + i] = Real(1);
    }
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t p = k;
        for (std::size_t i = k + 1; i < n; ++i) {
            if (std::abs(a[i * n + k]) > std::abs(a[p * n + k])) {
                p = i;
            }
        }
        FE_THROW_IF(a[p * n + k] == Real(0), InvalidStateException,
                    "HDivNormalConstraint: singular face orientation transform");
        for (std::size_t j = 0; j < n; ++j) {
            std::swap(a[k * n + j], a[p * n + j]);
            std::swap(inv[k * n + j], inv[p * n + j]);
        }
        const Real pivot = a[k * n + k];
        for (std::size_t j = 0; j < n; ++j) {
            a[k * n + j] /= pivot;
            inv[k * n + j] /= pivot;
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (i == k) {
                continue;
            }
            const Real f = a[i * n + k];
            for (std::size_t j = 0; j < n; ++j) {
                a[i * n + j] -= f * a[k * n + j];
                inv[i * n + j] -= f * inv[k * n + j];
            }
        }
    }
    return inv;
}

/// The global values of a face's DOFs from the cell's local ones, in the order
/// of @p local_face_dofs: the inverse of the transform the assembler applies
/// to a cell's coefficients (global to local) for the face's edge (2D) or face
/// (3D) DOFs.
[[nodiscard]] std::vector<Real> faceLocalToGlobal(const dofs::DofHandler& dof_handler,
                                                  const assembly::IMeshAccess& mesh,
                                                  GlobalIndex cell_id,
                                                  const elements::Element& element,
                                                  const std::vector<int>& local_face_dofs)
{
    const auto n = local_face_dofs.size();
    std::vector<Real> identity(n * n, Real(0));
    for (std::size_t i = 0; i < n; ++i) {
        identity[i * n + i] = Real(1);
    }
    if (!dof_handler.hasCellOrientations()) {
        // A single cell has the canonical orientation (the assembler's rule).
        FE_THROW_IF(mesh.numCells() > 1, InvalidStateException,
                    "HDivNormalConstraint: the face orientations of a mesh of several cells need a DofHandler "
                    "with cell orientations");
        return identity;
    }

    const auto* vbf = dynamic_cast<const basis::VectorBasisFunction*>(&element.basis());
    FE_CHECK_NOT_NULL(vbf, "HDivNormalConstraint: vector basis");
    const auto associations = vbf->dof_associations();

    // The face DOFs by moment index, as the assembler orders them.
    std::vector<std::pair<int, std::size_t>> by_moment;
    by_moment.reserve(n);
    basis::DofEntity entity = basis::DofEntity::Interior;
    int entity_id = -1;
    for (std::size_t k = 0; k < n; ++k) {
        const auto& a = associations.at(static_cast<std::size_t>(local_face_dofs[k]));
        FE_THROW_IF(entity_id >= 0 && (a.entity_type != entity || a.entity_id != entity_id), InvalidStateException,
                    "HDivNormalConstraint: the DOFs of a face belong to more than one entity");
        entity = a.entity_type;
        entity_id = a.entity_id;
        by_moment.emplace_back(a.moment_index, k);
    }
    std::sort(by_moment.begin(), by_moment.end());

    // T: global to local, in moment order (local = T global).
    std::vector<Real> T(n * n, Real(0));
    std::vector<Real> unit(n, Real(0));
    const ElementType cell_type = mesh.getCellType(cell_id);
    if (entity == basis::DofEntity::Edge) {
        const auto orientations = dof_handler.cellEdgeOrientations(cell_id);
        FE_THROW_IF(entity_id < 0 || static_cast<std::size_t>(entity_id) >= orientations.size(), InvalidStateException,
                    "HDivNormalConstraint: missing the cell's edge orientation");
        for (std::size_t c = 0; c < n; ++c) {
            std::fill(unit.begin(), unit.end(), Real(0));
            unit[c] = Real(1);
            const auto column = spaces::OrientationManager::orient_hcurl_edge_dofs(
                unit, orientations[static_cast<std::size_t>(entity_id)]);
            for (std::size_t r = 0; r < n; ++r) {
                T[r * n + c] = column[r];
            }
        }
    } else if (entity == basis::DofEntity::Face) {
        const auto orientations = dof_handler.cellFaceOrientations(cell_id);
        FE_THROW_IF(entity_id < 0 || static_cast<std::size_t>(entity_id) >= orientations.size(), InvalidStateException,
                    "HDivNormalConstraint: missing the cell's face orientation");
        const auto& orientation = orientations[static_cast<std::size_t>(entity_id)];
        const auto& ref = elements::ReferenceElement::shared(cell_type);
        const bool triangle = ref.face_nodes(static_cast<std::size_t>(entity_id)).size() == 3u;
        // The assembler's transform is the inverse of the orientation map O.
        std::vector<Real> O(n * n, Real(0));
        for (std::size_t c = 0; c < n; ++c) {
            std::fill(unit.begin(), unit.end(), Real(0));
            unit[c] = Real(1);
            const auto column = triangle
                ? spaces::OrientationManager::orient_triangle_face_dofs(unit, orientation, element.polynomial_order())
                : spaces::OrientationManager::orient_quad_face_dofs(unit, orientation, element.polynomial_order());
            for (std::size_t r = 0; r < n; ++r) {
                O[r * n + c] = column[r];
            }
        }
        T = invertSmallMatrix(std::move(O), n);
    } else {
        FE_THROW(InvalidStateException, "HDivNormalConstraint: a face DOF is not on an edge or a face");
    }

    // local_to_global = T^{-1}, from moment order back to local_face_dofs order.
    const auto inverse = invertSmallMatrix(std::move(T), n);
    std::vector<Real> out(n * n, Real(0));
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            out[by_moment[r].second * n + by_moment[c].second] = inverse[r * n + c];
        }
    }
    return out;
}

/// The factor relating the normal trace of a reference function to the
/// physical one: v.n_ref = g det(J) |J^-T n_ref| makes the Piola image's
/// outward normal trace g (the cofactor matrix det(J) J^-T maps n_ref).
[[nodiscard]] Real normalTraceScale(const math::Matrix<Real, 3, 3>& J, const spaces::FunctionSpace::Value& n_ref, int dim)
{
    if (dim == 2) {
        const Real det = J(0, 0) * J(1, 1) - J(0, 1) * J(1, 0);
        const Real c0 = J(1, 1) * n_ref[0] - J(1, 0) * n_ref[1];
        const Real c1 = -J(0, 1) * n_ref[0] + J(0, 0) * n_ref[1];
        return std::copysign(std::sqrt(c0 * c0 + c1 * c1), det);
    }
    const math::Vector<Real, 3> a{J(0, 0), J(1, 0), J(2, 0)};
    const math::Vector<Real, 3> b{J(0, 1), J(1, 1), J(2, 1)};
    const math::Vector<Real, 3> c{J(0, 2), J(1, 2), J(2, 2)};
    const auto bc = b.cross(c);
    const auto ca = c.cross(a);
    const auto ab = a.cross(b);
    const math::Vector<Real, 3> cof_n{bc[0] * n_ref[0] + ca[0] * n_ref[1] + ab[0] * n_ref[2],
                                      bc[1] * n_ref[0] + ca[1] * n_ref[1] + ab[1] * n_ref[2],
                                      bc[2] * n_ref[0] + ca[2] * n_ref[1] + ab[2] * n_ref[2]};
    const Real det = a.dot(bc);
    return std::copysign(std::sqrt(cof_n.dot(cof_n)), det);
}

} // namespace

void HDivNormalConstraint::buildFaceWorkItems_(const FESystem& system)
{
    const auto& rec = system.fieldRecord(field_);
    FE_CHECK_NOT_NULL(rec.space.get(), "HDivNormalConstraint: field space");

    FE_THROW_IF(rec.space->continuity() != Continuity::H_div, InvalidArgumentException,
                "HDivNormalConstraint requires an H(div) space");
    FE_THROW_IF(rec.space->field_type() != FieldType::Vector, InvalidArgumentException,
                "HDivNormalConstraint requires a vector-valued field");
    FE_THROW_IF(!rec.space->element().basis().is_vector_valued(), InvalidArgumentException,
                "HDivNormalConstraint requires a vector-basis space");

    const auto volume_space = std::const_pointer_cast<spaces::FunctionSpace>(rec.space);
    FE_CHECK_NOT_NULL(volume_space.get(), "HDivNormalConstraint: volume space");

    const auto& mesh = system.meshAccess();
    const auto& dh = system.fieldDofHandler(field_);
    const auto& dof_map = dh.getDofMap();
    const GlobalIndex offset = system.fieldDofOffset(field_);

    face_work_items_.clear();
    dofs_.clear();

    std::vector<std::array<Real, 3>> cell_coords;
    std::vector<math::Vector<Real, 3>> geom_nodes;

    mesh.forEachBoundaryFace(boundary_marker_, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        const auto cell_dofs = dof_map.getCellDofs(cell_id);
        FE_THROW_IF(cell_dofs.empty(), InvalidStateException,
                    "HDivNormalConstraint: empty cell DOF list");

        const auto local_face = static_cast<int>(mesh.getLocalFaceIndex(face_id, cell_id));
        auto trace_space = std::make_shared<spaces::TraceSpace>(volume_space, local_face);
        const auto local_face_dofs = trace_space->face_dof_indices();

        cell_coords.clear();
        mesh.getCellCoordinates(cell_id, cell_coords);
        FE_THROW_IF(cell_coords.empty(), InvalidStateException,
                    "HDivNormalConstraint: empty cell coordinate list");

        geom_nodes.resize(cell_coords.size());
        for (std::size_t i = 0; i < cell_coords.size(); ++i) {
            geom_nodes[i][0] = cell_coords[i][0];
            geom_nodes[i][1] = cell_coords[i][1];
            geom_nodes[i][2] = cell_coords[i][2];
        }

        geometry::MappingRequest req;
        req.element_type = mesh.getCellType(cell_id);
        req.geometry_order = 1;
        req.use_affine = true;
        auto mapping = geometry::MappingFactory::create(req, geom_nodes);
        FE_CHECK_NOT_NULL(mapping.get(), "HDivNormalConstraint: cell mapping");

        FaceWorkItem item;
        const auto& element = rec.space->getElement(req.element_type, cell_id);
        item.dimension = mesh.dimension();
        item.basis = element.basis_ptr();
        item.local_face_dofs = local_face_dofs;
        item.local_to_global = faceLocalToGlobal(dh, mesh, cell_id, element, local_face_dofs);
        item.trace_space = std::move(trace_space);
        item.cell_mapping = std::move(mapping);
        item.global_dofs.reserve(local_face_dofs.size());
        for (int ldof : local_face_dofs) {
            FE_THROW_IF(ldof < 0, InvalidArgumentException,
                        "HDivNormalConstraint: negative local face DOF index");
            const auto ldof_u = static_cast<std::size_t>(ldof);
            FE_THROW_IF(ldof_u >= cell_dofs.size(), InvalidStateException,
                        "HDivNormalConstraint: local face DOF index out of range");
            const auto dof = cell_dofs[ldof_u] + offset;
            item.global_dofs.push_back(dof);
            dofs_.push_back(dof);
        }

        face_work_items_.push_back(std::move(item));
    });

    std::sort(dofs_.begin(), dofs_.end());
    dofs_.erase(std::unique(dofs_.begin(), dofs_.end()), dofs_.end());
}

std::vector<Real> HDivNormalConstraint::faceValues_(const FaceWorkItem& item,
                                                    double time,
                                                    double dt) const
{
    FE_CHECK_NOT_NULL(item.trace_space.get(), "HDivNormalConstraint: trace space");
    FE_CHECK_NOT_NULL(item.cell_mapping.get(), "HDivNormalConstraint: cell mapping");

    forms::PointEvalContext pctx;
    pctx.time = static_cast<Real>(time);
    pctx.dt = static_cast<Real>(dt);

    // The cell's local face coefficients: the L2 projection, on the reference
    // face, of the reference normal trace whose Piola image has outward normal
    // trace g onto the normal traces of the face's functions.
    const auto& n_ref = item.trace_space->face_normal();
    const auto n = item.local_face_dofs.size();
    const auto face_rule = quadrature::QuadratureFactory::create(
        item.trace_space->element_type(), 2 * item.basis->order() + 4);
    FE_CHECK_NOT_NULL(face_rule.get(), "HDivNormalConstraint: face quadrature");
    std::vector<Real> mass(n * n, Real(0));
    std::vector<Real> local_values(n, Real(0));
    std::vector<Real> traces(n, Real(0));
    std::vector<math::Vector<Real, 3>> values;
    for (std::size_t q = 0; q < face_rule->num_points(); ++q) {
        const auto xi_volume = item.trace_space->embed_face_point(face_rule->point(q));
        const auto x = item.cell_mapping->map_to_physical(xi_volume);
        pctx.x = {x[0], x[1], x[2]};
        const Real target = forms::evaluateScalarAt(value_, pctx) *
                            normalTraceScale(item.cell_mapping->jacobian(xi_volume), n_ref, item.dimension);
        item.basis->evaluate_vector_values(xi_volume, values);
        for (std::size_t i = 0; i < n; ++i) {
            const auto& v = values.at(static_cast<std::size_t>(item.local_face_dofs[i]));
            traces[i] = v[0] * n_ref[0] + v[1] * n_ref[1] + v[2] * n_ref[2];
        }
        const Real w = face_rule->weight(q);
        for (std::size_t i = 0; i < n; ++i) {
            local_values[i] += w * traces[i] * target;
            for (std::size_t j = 0; j < n; ++j) {
                mass[i * n + j] += w * traces[i] * traces[j];
            }
        }
    }
    const auto inverse_mass = invertSmallMatrix(std::move(mass), n);
    {
        std::vector<Real> projected(n, Real(0));
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                projected[i] += inverse_mass[i * n + j] * local_values[j];
            }
        }
        local_values = std::move(projected);
    }

    // The global DOF values: the face's orientation in the cell.
    FE_THROW_IF(item.local_to_global.size() != n * n, InvalidStateException,
                "HDivNormalConstraint: face orientation transform size mismatch");
    std::vector<Real> face_values(n, Real(0));
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            face_values[r] += item.local_to_global[r * n + c] * local_values[c];
        }
    }
    return face_values;
}

void HDivNormalConstraint::apply(const FESystem& system, constraints::AffineConstraints& constraints)
{
    if (value_.hasTest() || value_.hasTrial()) {
        throw std::invalid_argument("HDivNormalConstraint: value expression must not contain test/trial functions");
    }

    buildFaceWorkItems_(system);

    const auto& owned = system.dofHandler().getPartition().locallyOwned();
    for (const auto& item : face_work_items_) {
        const auto face_values = faceValues_(item, /*time=*/0.0, /*dt=*/0.0);
        FE_THROW_IF(face_values.size() != item.global_dofs.size(), InvalidStateException,
                    "HDivNormalConstraint: face value / DOF size mismatch");

        for (std::size_t i = 0; i < item.global_dofs.size(); ++i) {
            const auto dof = item.global_dofs[i];
            if (!owned.contains(dof)) {
                continue;
            }
            constraints.addDirichlet(dof, face_values[i]);
        }
    }
}

bool HDivNormalConstraint::updateValues(const FESystem& system,
                                        constraints::AffineConstraints& constraints,
                                        double time,
                                        double dt)
{
    if (!is_time_dependent_) {
        return false;
    }

    if (face_work_items_.empty()) {
        buildFaceWorkItems_(system);
    }

    const auto& owned = system.dofHandler().getPartition().locallyOwned();
    for (const auto& item : face_work_items_) {
        const auto face_values = faceValues_(item, time, dt);
        FE_THROW_IF(face_values.size() != item.global_dofs.size(), InvalidStateException,
                    "HDivNormalConstraint: face value / DOF size mismatch");

        for (std::size_t i = 0; i < item.global_dofs.size(); ++i) {
            const auto dof = item.global_dofs[i];
            if (!owned.contains(dof)) {
                continue;
            }
            constraints.updateInhomogeneity(dof, face_values[i]);
        }
    }

    return true;
}

} // namespace constraints
} // namespace FE
} // namespace svmp

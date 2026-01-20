/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/StrongDirichletConstraint.h"

#include "Systems/FESystem.h"
#include "Systems/SystemsExceptions.h"

#include "Forms/PointEvaluator.h"

#include "Geometry/MappingFactory.h"
#include "Spaces/FaceRestriction.h"

#include <algorithm>
#include <unordered_map>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

namespace {

struct BoundaryDofsWithCoords {
    std::vector<GlobalIndex> dofs;
    std::vector<std::array<Real, 3>> coords;
};

[[nodiscard]] spaces::FaceRestriction& getFaceRestriction(
    std::unordered_map<int, spaces::FaceRestriction>& cache,
    ElementType element_type,
    int order,
    Continuity continuity)
{
    const int key = static_cast<int>(element_type);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    auto [ins, inserted] = cache.emplace(key, spaces::FaceRestriction(element_type, order, continuity));
    (void)inserted;
    return ins->second;
}

[[nodiscard]] BoundaryDofsWithCoords boundaryDofsWithCoordsByMarker(
    const FESystem& system,
    FieldId field,
    int boundary_marker,
    int component)
{
    BoundaryDofsWithCoords out;

    const auto& rec = system.fieldRecord(field);
    FE_CHECK_NOT_NULL(rec.space.get(), "StrongDirichletConstraint: field space");

    const auto& mesh = system.meshAccess();
    const auto& dh = system.fieldDofHandler(field);
    const auto& dof_map = dh.getDofMap();
    const GlobalIndex offset = system.fieldDofOffset(field);

    const int n_components = std::max(1, rec.space->value_dimension());
    if (component < -1) {
        throw std::invalid_argument("StrongDirichletConstraint: component must be >= -1");
    }
    if (component >= n_components) {
        throw std::invalid_argument("StrongDirichletConstraint: component index out of range");
    }

    std::unordered_map<GlobalIndex, std::array<Real, 3>> coord_by_dof;
    std::unordered_map<int, spaces::FaceRestriction> face_restriction_cache;

    std::vector<std::array<Real, 3>> cell_coords;
    std::vector<math::Vector<Real, 3>> geom_nodes;

    mesh.forEachBoundaryFace(boundary_marker, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        const ElementType cell_type = mesh.getCellType(cell_id);

        auto& face_restriction = getFaceRestriction(face_restriction_cache,
                                                    cell_type,
                                                    rec.space->polynomial_order(),
                                                    rec.space->continuity());

        const auto face_local = static_cast<int>(mesh.getLocalFaceIndex(face_id, cell_id));
        const auto local_face_dofs = face_restriction.face_dofs(face_local);

        const auto cell_dofs = dof_map.getCellDofs(cell_id);
        FE_THROW_IF(cell_dofs.empty(), InvalidStateException,
                    "StrongDirichletConstraint: empty cell DOF list");

        const auto dofs_per_component = cell_dofs.size() / static_cast<std::size_t>(n_components);
        FE_THROW_IF(dofs_per_component == 0 || dofs_per_component * static_cast<std::size_t>(n_components) != cell_dofs.size(),
                    InvalidStateException,
                    "StrongDirichletConstraint: cell DOF list size is not divisible by field components");

        cell_coords.clear();
        mesh.getCellCoordinates(cell_id, cell_coords);
        FE_THROW_IF(cell_coords.empty(), InvalidStateException,
                    "StrongDirichletConstraint: empty cell coordinate list");

        geom_nodes.resize(cell_coords.size());
        for (std::size_t i = 0; i < cell_coords.size(); ++i) {
            geom_nodes[i][0] = cell_coords[i][0];
            geom_nodes[i][1] = cell_coords[i][1];
            geom_nodes[i][2] = cell_coords[i][2];
        }

        geometry::MappingRequest req;
        req.element_type = cell_type;
        req.geometry_order = 1;
        req.use_affine = true;
        auto mapping = geometry::MappingFactory::create(req, geom_nodes);
        FE_CHECK_NOT_NULL(mapping.get(), "StrongDirichletConstraint: mapping");

        const auto& dof_nodes = face_restriction.dof_nodes();

        for (int ldof : local_face_dofs) {
            FE_THROW_IF(ldof < 0, InvalidArgumentException,
                        "StrongDirichletConstraint: negative local DOF index");
            const auto ldof_u = static_cast<std::size_t>(ldof);
            FE_THROW_IF(ldof_u >= dofs_per_component, InvalidArgumentException,
                        "StrongDirichletConstraint: local DOF index out of range for cell DOF list");
            FE_THROW_IF(ldof_u >= dof_nodes.size(), InvalidArgumentException,
                        "StrongDirichletConstraint: local DOF index out of range for dof_nodes");

            const auto x = mapping->map_to_physical(dof_nodes[ldof_u]);

            const int comp_begin = (component < 0) ? 0 : component;
            const int comp_end = (component < 0) ? n_components : (component + 1);
            for (int comp = comp_begin; comp < comp_end; ++comp) {
                const auto local = static_cast<std::size_t>(comp) * dofs_per_component + ldof_u;
                FE_THROW_IF(local >= cell_dofs.size(), InvalidStateException,
                            "StrongDirichletConstraint: component-local DOF index out of range");

                const GlobalIndex dof = cell_dofs[local] + offset;
                if (coord_by_dof.find(dof) != coord_by_dof.end()) {
                    continue;
                }
                coord_by_dof.emplace(dof, std::array<Real, 3>{x[0], x[1], x[2]});
            }
        }
    });

    out.dofs.reserve(coord_by_dof.size());
    for (const auto& [dof, _] : coord_by_dof) {
        out.dofs.push_back(dof);
    }
    std::sort(out.dofs.begin(), out.dofs.end());
    out.dofs.erase(std::unique(out.dofs.begin(), out.dofs.end()), out.dofs.end());

    out.coords.reserve(out.dofs.size());
    for (const auto dof : out.dofs) {
        auto it = coord_by_dof.find(dof);
        FE_THROW_IF(it == coord_by_dof.end(), InvalidStateException,
                    "StrongDirichletConstraint: internal DOF->coord map missing key");
        out.coords.push_back(it->second);
    }

    return out;
}

} // namespace

StrongDirichletConstraint::StrongDirichletConstraint(FieldId field, int boundary_marker, forms::FormExpr value, int component)
    : field_(field),
      boundary_marker_(boundary_marker),
      component_(component),
      value_(std::move(value))
{
    if (field_ == INVALID_FIELD_ID) {
        throw std::invalid_argument("StrongDirichletConstraint: invalid FieldId");
    }
    if (component_ < -1) {
        throw std::invalid_argument("StrongDirichletConstraint: component must be >= -1");
    }
    if (!value_.isValid()) {
        throw std::invalid_argument("StrongDirichletConstraint: invalid value expression");
    }
    is_time_dependent_ = forms::isTimeDependent(value_);
}

void StrongDirichletConstraint::apply(const FESystem& system, constraints::AffineConstraints& constraints)
{
    if (value_.hasTest() || value_.hasTrial()) {
        throw std::invalid_argument("StrongDirichletConstraint: value expression must not contain test/trial functions");
    }

    auto extracted = boundaryDofsWithCoordsByMarker(system, field_, boundary_marker_, component_);
    dofs_ = std::move(extracted.dofs);
    coords_ = std::move(extracted.coords);

    FE_THROW_IF(dofs_.size() != coords_.size(), InvalidStateException,
                "StrongDirichletConstraint: DOF/coordinate size mismatch");

    const auto& owned = system.dofHandler().getPartition().locallyOwned();

    forms::PointEvalContext pctx;
    pctx.time = 0.0;
    pctx.dt = 0.0;

    for (std::size_t i = 0; i < dofs_.size(); ++i) {
        const auto dof = dofs_[i];
        if (!owned.contains(dof)) {
            continue;
        }
        pctx.x = coords_[i];
        const Real v = forms::evaluateScalarAt(value_, pctx);
        constraints.addLine(dof);
        constraints.setInhomogeneity(dof, v);
    }
}

bool StrongDirichletConstraint::updateValues(const FESystem& /*system*/,
                                             constraints::AffineConstraints& constraints,
                                             double time,
                                             double dt)
{
    if (!is_time_dependent_) {
        return false;
    }

    forms::PointEvalContext pctx;
    pctx.time = static_cast<Real>(time);
    pctx.dt = static_cast<Real>(dt);

    for (std::size_t i = 0; i < dofs_.size(); ++i) {
        pctx.x = coords_[i];
        const Real v = forms::evaluateScalarAt(value_, pctx);
        constraints.updateInhomogeneity(dofs_[i], v);
    }

    return true;
}

} // namespace systems
} // namespace FE
} // namespace svmp

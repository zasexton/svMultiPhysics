/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/HDivNormalConstraint.h"

#include "Forms/PointEvaluator.h"
#include "Geometry/MappingFactory.h"
#include "Spaces/TraceSpace.h"
#include "Systems/FESystem.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <array>
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

    std::vector<Real> face_values;
    item.trace_space->interpolate(
        [&](const spaces::FunctionSpace::Value& xi_face) -> spaces::FunctionSpace::Value {
            const auto xi_volume = item.trace_space->embed_face_point(xi_face);
            const auto x = item.cell_mapping->map_to_physical(xi_volume);
            pctx.x = {x[0], x[1], x[2]};

            spaces::FunctionSpace::Value out{};
            out[0] = forms::evaluateScalarAt(value_, pctx);
            return out;
        },
        face_values);

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

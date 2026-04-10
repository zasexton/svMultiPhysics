/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/AuxiliaryDrivenDirichletConstraint.h"

#include "Systems/FESystem.h"
#include "Systems/SystemsExceptions.h"

#include "Geometry/MappingFactory.h"
#include "Spaces/FaceRestriction.h"

#include <algorithm>
#include <unordered_map>
#include <utility>

namespace svmp {
namespace FE {
namespace constraints {

using systems::FESystem;
using systems::InvalidStateException;

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
    auto [ins, inserted] = cache.emplace(
        key,
        spaces::FaceRestriction(element_type, order, continuity));
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
    FE_CHECK_NOT_NULL(rec.space.get(), "AuxiliaryDrivenDirichletConstraint: field space");
    FE_THROW_IF(rec.space->continuity() == Continuity::H_curl ||
                    rec.space->continuity() == Continuity::H_div ||
                    rec.space->element().basis().is_vector_valued(),
                InvalidArgumentException,
                "AuxiliaryDrivenDirichletConstraint does not support H(curl)/H(div) "
                "vector-basis spaces; use the dedicated vector-basis constraint types instead");

    const auto& mesh = system.meshAccess();
    const auto& dh = system.fieldDofHandler(field);
    const auto& dof_map = dh.getDofMap();
    const GlobalIndex offset = system.fieldDofOffset(field);

    const int n_components = std::max(1, rec.space->value_dimension());
    FE_THROW_IF(component < -1, InvalidArgumentException,
                "AuxiliaryDrivenDirichletConstraint: component must be >= -1");
    FE_THROW_IF(component >= n_components, InvalidArgumentException,
                "AuxiliaryDrivenDirichletConstraint: component index out of range");

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
                    "AuxiliaryDrivenDirichletConstraint: empty cell DOF list");

        const auto dofs_per_component =
            cell_dofs.size() / static_cast<std::size_t>(n_components);
        FE_THROW_IF(dofs_per_component == 0 ||
                        dofs_per_component * static_cast<std::size_t>(n_components) !=
                            cell_dofs.size(),
                    InvalidStateException,
                    "AuxiliaryDrivenDirichletConstraint: cell DOF list size is not "
                    "divisible by field components");

        cell_coords.clear();
        mesh.getCellCoordinates(cell_id, cell_coords);
        FE_THROW_IF(cell_coords.empty(), InvalidStateException,
                    "AuxiliaryDrivenDirichletConstraint: empty cell coordinate list");

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
        FE_CHECK_NOT_NULL(mapping.get(), "AuxiliaryDrivenDirichletConstraint: mapping");

        const auto& dof_nodes = face_restriction.dof_nodes();
        for (int ldof : local_face_dofs) {
            FE_THROW_IF(ldof < 0, InvalidArgumentException,
                        "AuxiliaryDrivenDirichletConstraint: negative local DOF index");
            const auto ldof_u = static_cast<std::size_t>(ldof);
            FE_THROW_IF(ldof_u >= dofs_per_component || ldof_u >= dof_nodes.size(),
                        InvalidArgumentException,
                        "AuxiliaryDrivenDirichletConstraint: local DOF index out of range");

            const auto x = mapping->map_to_physical(dof_nodes[ldof_u]);
            const int comp_begin = (component < 0) ? 0 : component;
            const int comp_end = (component < 0) ? n_components : (component + 1);
            for (int comp = comp_begin; comp < comp_end; ++comp) {
                const auto local = static_cast<std::size_t>(comp) * dofs_per_component + ldof_u;
                FE_THROW_IF(local >= cell_dofs.size(), InvalidStateException,
                            "AuxiliaryDrivenDirichletConstraint: component-local DOF index out of range");
                const GlobalIndex dof = cell_dofs[local] + offset;
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
                    "AuxiliaryDrivenDirichletConstraint: internal DOF/coord map missing key");
        out.coords.push_back(it->second);
    }

    return out;
}

[[nodiscard]] int parseBoundaryMarker(const systems::AuxiliaryConstraintBinding& binding,
                                      std::string_view instance_name)
{
    FE_THROW_IF(binding.target_region.kind != systems::AuxiliaryRegionKind::BoundarySet,
                InvalidArgumentException,
                "AuxiliaryDrivenDirichletConstraint: instance '" + std::string(instance_name) +
                    "' strong-Dirichlet bindings must target BoundarySet regions");
    try {
        return std::stoi(binding.target_region.identity);
    } catch (const std::exception&) {
        FE_THROW(InvalidArgumentException,
                 "AuxiliaryDrivenDirichletConstraint: instance '" +
                     std::string(instance_name) +
                     "' has non-integer boundary marker '" +
                     binding.target_region.identity + "'");
    }
}

} // namespace

AuxiliaryDrivenDirichletConstraint::AuxiliaryDrivenDirichletConstraint(
    std::string instance_name,
    systems::AuxiliaryConstraintBinding binding)
    : instance_name_(std::move(instance_name))
    , binding_(std::move(binding))
{
    FE_THROW_IF(instance_name_.empty(), InvalidArgumentException,
                "AuxiliaryDrivenDirichletConstraint: empty instance name");
    FE_THROW_IF(binding_.target_field == INVALID_FIELD_ID, InvalidArgumentException,
                "AuxiliaryDrivenDirichletConstraint: invalid target field");
}

void AuxiliaryDrivenDirichletConstraint::apply(const FESystem& system,
                                               AffineConstraints& constraints)
{
    const int boundary_marker = parseBoundaryMarker(binding_, instance_name_);

    auto extracted = boundaryDofsWithCoordsByMarker(
        system, binding_.target_field, boundary_marker, binding_.target_component);
    dofs_ = std::move(extracted.dofs);
    coords_ = std::move(extracted.coords);

    FE_THROW_IF(dofs_.size() != coords_.size(), InvalidStateException,
                "AuxiliaryDrivenDirichletConstraint: DOF/coordinate size mismatch");

    const auto& owned = system.dofHandler().getPartition().locallyOwned();
    const Real value = system.auxiliaryConstraintValue(
        instance_name_, binding_, /*time=*/Real{0.0}, /*dt=*/Real{0.0});

    for (const auto dof : dofs_) {
        if (!owned.contains(dof)) {
            continue;
        }
        constraints.addDirichlet(dof, value);
    }
}

bool AuxiliaryDrivenDirichletConstraint::updateValues(const FESystem& system,
                                                      AffineConstraints& constraints,
                                                      double time,
                                                      double dt)
{
    const Real value = system.auxiliaryConstraintValue(
        instance_name_, binding_, static_cast<Real>(time), static_cast<Real>(dt));
    for (const auto dof : dofs_) {
        constraints.updateInhomogeneity(dof, value);
    }
    return true;
}

} // namespace constraints
} // namespace FE
} // namespace svmp
